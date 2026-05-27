"""
main.py — GRPO 完整训练流程
================================
四个阶段：
  Phase 1 (rollout): 加载 prompts → Tokenize → 扩展 B→B*G → 生成 response
  Phase 2 (reward):  计算标量奖励 → 组内标准化为 Advantage
  Phase 3 (loss):    Actor/Ref 前向 → log-probs → ratio & KL → GRPO 策略损失
  Phase 4 (train):   backward + optimizer.step + 周期性日志/保存
"""

import os
import json
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import random

# 导入我们自己写的三个阶段模块
import sys
sys.path.insert(0, "src")
from rollout import rollout
from reward import compute_rewards_and_advantages
from loss import compute_loss


# ============================================================
# 配置区
# ============================================================

CONFIG = {
    # --- 模型 ---
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",

    # --- Rollout 参数 ---
    "prompt_file":   "data/prompts.jsonl",
    "batch_size": 4,
    "G":             8,              # 每个 prompt 生成 G 个 response
    "max_new_tokens": 300,           # 最大生成 token 数
    "temperature":    0.8,           # 采样温度

    # --- Loss 参数 ---
    "eps_clip": 0.2,                 # PPO clip 范围
    "beta":     0.04,                # KL 惩罚系数

    # --- 训练参数 ---
    "num_epochs":          3,        # 训练 epoch 数
    "steps_per_epoch":     10,       # 每个 epoch 做多少次 rollout+update
    "learning_rate":       5e-6,     # Actor 的学习率
    "lr_scheduler_gamma":  0.9,      # 每个 epoch 后的衰减率
    "grad_clip_norm":      1.0,      # 梯度裁剪最大 norm

    # --- 日志与保存 ---
    "log_interval":  1,              # 每 N 步打印详细日志
    "save_dir":      "checkpoints",  # 模型保存目录
    "save_interval": 10,              # 每 N 步保存一次 checkpoint

    # --- 资源 ---
    "dtype": torch.bfloat16,         # 模型精度
}


# ============================================================
# 工具函数
# ============================================================

def setup_model_and_tokenizer(config: dict):
    """
    功能：加载 Actor 模型、Reference 模型和 Tokenizer

    为什么需要两个模型？
      Actor (可训练):  通过梯度下降更新策略
      Reference (冻结):  作为 KL 散度的参考基准，防止策略偏离太远

    返回：
        model, ref_model, tokenizer
    """
    model_name = config["model_name"]

    print("=" * 60)
    print("[main] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Qwen2.5 默认没有 pad_token，手动设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[main] pad_token_id = {tokenizer.pad_token_id}")
    print(f"[main] eos_token_id = {tokenizer.eos_token_id}")

    print("[main] Loading Actor model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=config["dtype"],
        device_map="auto",
    )

    print("[main] Loading Reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=config["dtype"],
        device_map="auto",
    )

    # Reference 模型冻结所有参数
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] Total params: {total_params/1e6:.1f}M, "
          f"Actor trainable: {trainable_params/1e6:.1f}M")
    print(f"[main] Device: {next(model.parameters()).device}")
    print("=" * 60)

    return model, ref_model, tokenizer


def setup_optimizer(model, config: dict):
    """
    功能：设置 AdamW 优化器 + 学习率调度器

    为什么需要 lr_scheduler？
      GRPO 训练中，随着策略逐渐优化，需要更精细的更新。
      逐步衰减学习率有助于训练稳定性。
    """
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,                       # 每个 epoch 衰减
        gamma=config["lr_scheduler_gamma"],
    )

    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, step: int, metrics: dict, config: dict):
    """
    功能：保存训练 checkpoint

    保存内容：
      - Actor 模型权重 (pytorch_model.bin)
      - optimizer 状态
      - scheduler 状态
      - 当前 step 数
      - 最近的 metrics
    """
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 保存完整 checkpoint
    checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }, checkpoint_path)

    # 也单独保存一份可直接加载的模型权重（HuggingFace 格式）
    hf_save_dir = os.path.join(save_dir, f"model_step_{step}")
    model.save_pretrained(hf_save_dir)

    print(f"[checkpoint] Saved checkpoint to {checkpoint_path}")
    print(f"[checkpoint] Saved HF model to {hf_save_dir}")


def log_metrics(step: int, metrics: dict, epoch: int):
    """
    功能：格式化输出训练指标
    """
    print(f"\n{'=' * 60}")
    print(f"[Step {step}] (Epoch {epoch}) Metrics Summary:")
    print(f"{'=' * 60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {key:25s}: {value}")
        else:
            print(f"  {key:25s}: {value}")
    print(f"{'=' * 60}\n")


def clear_cuda_cache(step: int, config: dict):
    """
    功能：定期清理 GPU 缓存，防止显存碎片化

    在每步结束后调用。Rollout 产生大量中间张量，
    batch dict 重置后清理缓存有助于长期训练稳定。
    """
    # 每 5 步做一次完整清理
    if step % 5 == 0:
        torch.cuda.empty_cache()


# ============================================================
# 主训练循环
# ============================================================

def main():
    config = CONFIG
    print(f"\n{'#' * 60}")
    print(f"#  MinimalGRPO — 完整四阶段训练流程")
    print(f"#  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}\n")

    # 一次性加载全部 prompts 到内存
    all_prompts = []
    with open(config["prompt_file"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_prompts.append(json.loads(line)["prompt"])
    print(f"[main] Loaded {len(all_prompts)} total prompts, "
          f"batch_size={config['batch_size']}")

    # --- 初始化 ---
    model, ref_model, tokenizer = setup_model_and_tokenizer(config)
    optimizer, scheduler = setup_optimizer(model, config)

    # 训练状态
    global_step = 0
    total_rewards_history = []    # 记录所有 reward 均值
    total_advantages_history = []  # 记录所有 advantage 统计
    loss_history = []              # 记录所有 loss 值
    kl_history = []                # 记录所有 KL 均值

    print(f"\n[main] Starting training loop...")
    print(f"[main] Epochs: {config['num_epochs']}, "
          f"Steps per epoch: {config['steps_per_epoch']}")
    print(f"[main] Total steps: {config['num_epochs'] * config['steps_per_epoch']}")

    # --- 训练循环 ---
    for epoch in range(config["num_epochs"]):
        print(f"\n{'#' * 60}")
        print(f"#  EPOCH {epoch + 1}/{config['num_epochs']}")
        print(f"#  LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'#' * 60}")

        for step_in_epoch in range(config["steps_per_epoch"]):
            global_step += 1
            print(f"\n{'─' * 60}")
            print(f"[Step {global_step}] (Epoch {epoch+1}, "
                  f"Step {step_in_epoch+1}/{config['steps_per_epoch']})")
            print(f"{'─' * 60}")

            # ============================================
            # Phase 1: Rollout — 采样生成
            # ============================================
            print(f"\n[Phase 1/4] Rollout...")

            # --- 随机采样 B=batch_size 条 prompts ---
            batch_prompts = random.sample(
                all_prompts, 
                min(config["batch_size"], len(all_prompts))
            )
            batch = rollout(
                model=model,
                tokenizer=tokenizer,
                prompts_list=batch_prompts,   # <-- 用这个替代 prompt_file
                G=config["G"],
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
            )
            B = len(batch["prompts_text"])
            G = config["G"]
            print(f"[Phase 1/4] Done. Generated {B * G} responses from {B} prompts.")

            # ============================================
            # Phase 2: Reward & Advantage
            # ============================================
            print(f"\n[Phase 2/4] Computing rewards & advantages...")
            batch = compute_rewards_and_advantages(batch)

            rewards = batch["rewards"]           # [B*G]
            advantages = batch["advantages"]      # [B*G]

            total_rewards_history.append(rewards.mean().item())
            total_advantages_history.append({
                "mean_adv": advantages.mean().item(),
                "std_adv": advantages.std().item(),
            })
            print(f"[Phase 2/4] Done.")

            # ============================================
            # Phase 3: Loss 计算
            # ============================================
            print(f"\n[Phase 3/4] Computing GRPO loss...")
            optimizer.zero_grad()          # 清空上次梯度
            batch = compute_loss(
                model=model,
                ref_model=ref_model,
                batch=batch,
                eps_clip=config["eps_clip"],
                beta=config["beta"],
            )

            loss = batch["loss"]            # 标量
            kl_per_token = batch["kl_per_token"]  # [B*G, L_response]
            loss_history.append(loss.item())

            # 计算 KL 均值（仅有效 token）
            mask = batch["response_mask"].bool()
            valid_kl = (kl_per_token * batch["response_mask"]).sum() / (mask.sum() + 1e-8)
            kl_history.append(valid_kl.item())
            print(f"[Phase 3/4] Done. Loss={loss.item():.4f}")

            # ============================================
            # Phase 4: 梯度反向传播与参数更新
            # ============================================
            print(f"\n[Phase 4/4] Backward + Optimizer step...")
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config["grad_clip_norm"],
            )

            optimizer.step()
            print(f"[Phase 4/4] Done. Grad norm (clipped): {grad_norm:.4f}")

            # --- 日志输出 ---
            if global_step % config["log_interval"] == 0 or step_in_epoch == 0:
                metrics = {
                    "step":            global_step,
                    "epoch":           epoch + 1,
                    "lr":              scheduler.get_last_lr()[0],
                    "loss":            loss.item(),
                    "reward_mean":     rewards.mean().item(),
                    "reward_std":      rewards.std().item(),
                    "adv_mean":        advantages.mean().item(),
                    "adv_std":         advantages.std().item(),
                    "mean_kl":         valid_kl.item(),
                    "grad_norm":       grad_norm,
                    "num_samples":     B * G,
                }
                log_metrics(global_step, metrics, epoch + 1)

            # --- 保存 checkpoint ---
            if global_step % config["save_interval"] == 0:
                metrics_save = {
                    "loss":          loss.item(),
                    "reward_mean":   rewards.mean().item(),
                    "mean_kl":       valid_kl.item(),
                    "grad_norm":     grad_norm,
                }
                save_checkpoint(model, optimizer, scheduler, global_step, metrics_save, config)

            # --- 清理 ---
            clear_cuda_cache(global_step, config)

        # --- Epoch 结束：学习率衰减 ---
        scheduler.step()

    # ============================================
    # 训练结束
    # ============================================
    print(f"\n{'#' * 60}")
    print(f"#  Training Complete!")
    print(f"#  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Total steps: {global_step}")
    print(f"{'#' * 60}")

    # --- 最终总结 ---
    print(f"\n[Summary] Training history:")
    if len(loss_history) > 0:
        print(f"  Loss:  min={min(loss_history):.4f}, "
              f"max={max(loss_history):.4f}, "
              f"final={loss_history[-1]:.4f}")
    if len(total_rewards_history) > 0:
        print(f"  Reward: min={min(total_rewards_history):.4f}, "
              f"max={max(total_rewards_history):.4f}, "
              f"final={total_rewards_history[-1]:.4f}")
    if len(kl_history) > 0:
        print(f"  KL:     min={min(kl_history):.4f}, "
              f"max={max(kl_history):.4f}, "
              f"final={kl_history[-1]:.4f}")

    # --- 保存最终 checkpoint ---
    final_metrics = {
        "loss":   loss_history[-1] if loss_history else 0.0,
        "reward_mean": total_rewards_history[-1] if total_rewards_history else 0.0,
        "mean_kl": kl_history[-1] if kl_history else 0.0,
    }
    save_checkpoint(model, optimizer, scheduler, global_step, final_metrics, config)

    print(f"\n[main] Final model saved to {config['save_dir']}/")
    print(f"[main] Done!")


if __name__ == "__main__":
    main()