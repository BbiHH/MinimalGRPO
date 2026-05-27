"""
main.py — GRPO 完整训练流程 (字典收集 + 统一日志版)
=====================================================
- 四个阶段：Rollout → Reward → Loss 计算 → 梯度更新
- PPO 内层循环：一次 rollout 多次更新，配合 early stopping
- 所有指标统一存入 metrics 字典，每步结束时一次性 swanlab.log
"""

import os
import json
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import random

import swanlab

import sys
sys.path.insert(0, "src")
from rollout import rollout
from reward import compute_rewards_and_advantages
from loss import compute_loss


CONFIG = {
    # --- 模型 ---
    "model_name":         "Qwen/Qwen2.5-0.5B-Instruct",

    # --- Rollout ---
    "prompt_file":        "data/prompts.jsonl",
    "batch_size":         4,              # 每次采样的 prompt 数量
    "G":                  8,              # 每个 prompt 生成 G 条回复
    "max_new_tokens":     300,            # 最大生成长度
    "temperature":        0.8,            # 采样温度

    # --- Loss ---
    "eps_clip":           0.2,            # PPO clip 范围
    "beta":               0.04,           # KL 惩罚系数
    "ppo_epochs":         4,              # 每轮 rollout 的更新次数
    "target_kl":          0.02,           # KL 超过此值则提前停止本批更新

    # --- 训练步数 ---
    "num_epochs":         10,             # 外层 epoch 数（主要用于学习率衰减）
    "steps_per_epoch":    50,             # 每个 epoch 的 rollout 次数
    # 总 rollout 次数 = 10 * 50 = 500

    # --- 优化器 ---
    "learning_rate":      5e-6,
    "lr_scheduler_gamma": 0.9,           # 每个 epoch 衰减一次
    "grad_clip_norm":     1.0,           # 梯度裁剪最大 norm

    # --- 日志与保存 ---
    "log_interval":       1,              # 每 N 次 rollout 打印一次详细指标
    "save_dir":           "checkpoints",
    "save_interval":      100,            # 每 N 次 rollout 保存模型

    # --- 资源 ---
    "dtype":              torch.bfloat16,
}


def setup_model_and_tokenizer(config):
    """加载 Actor 模型、冻结的 Reference 模型和分词器

    Actor (可训练)：通过梯度下降更新策略
    Reference (冻结)：作为 KL 散度的参考基准，防止策略偏离太远
    """
    print("=" * 60)
    print("[main] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[main] pad_token_id = {tokenizer.pad_token_id}")

    print("[main] Loading Actor model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=config["dtype"], device_map="auto")
    print("[main] Loading Reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=config["dtype"], device_map="auto")
    # Reference 模型冻结所有参数
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] Total params: {total/1e6:.1f}M, Actor trainable: {trainable/1e6:.1f}M")
    print("=" * 60)
    return model, ref_model, tokenizer


def setup_optimizer(model, config):
    """AdamW 优化器 + 按 epoch 衰减的学习率调度器"""
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"],
                      betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=config["lr_scheduler_gamma"])
    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, step, metrics, config):
    """保存完整 checkpoint 和 HuggingFace 格式的模型权重"""
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    # 保存包含优化器状态的完整 checkpoint
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }, os.path.join(save_dir, f"checkpoint_step_{step}.pt"))
    # 保存可直接加载的 HuggingFace 模型
    model.save_pretrained(os.path.join(save_dir, f"model_step_{step}"))
    print(f"[checkpoint] Saved step {step}")


def log_metrics(step, metrics):
    """格式化打印关键训练指标"""
    print(f"\n{'='*60}")
    print(f"[Step {step}] Metrics Summary:")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        elif isinstance(v, int):
            print(f"  {k:25s}: {v}")
        else:
            print(f"  {k:25s}: {v}")
    print(f"{'='*60}\n")


def clear_cuda_cache(step):
    """定期清理显存碎片，避免长期训练时 OOM"""
    if step % 5 == 0:
        torch.cuda.empty_cache()


def main():
    config = CONFIG
    print(f"\n{'#'*60}\n#  MinimalGRPO — 字典收集 + 统一日志版\n"
          f"#  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'#'*60}\n")

    # 初始化 SwanLab 项目
    swanlab.init(project="MinimalGRPO", experiment_name="raw_1", config=config, mode="cloud")

    # 一次性加载所有 prompt
    all_prompts = []
    with open(config["prompt_file"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_prompts.append(json.loads(line)["prompt"])
    print(f"[main] Loaded {len(all_prompts)} prompts, batch_size={config['batch_size']}")

    # 初始化模型、优化器
    model, ref_model, tokenizer = setup_model_and_tokenizer(config)
    optimizer, scheduler = setup_optimizer(model, config)

    # 历史记录（用于最终总结）
    loss_history, reward_history, kl_history = [], [], []
    global_step = 0
    total_rollouts = config["num_epochs"] * config["steps_per_epoch"]
    print(f"[main] Total rollouts: {total_rollouts}\n")

    # ==============================
    # 训练主循环
    # ==============================
    for epoch in range(config["num_epochs"]):
        print(f"\n{'#'*60}\n#  EPOCH {epoch+1}/{config['num_epochs']}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}\n{'#'*60}")

        for _ in range(config["steps_per_epoch"]):
            global_step += 1

            # ============================================
            # Phase 1: Rollout — 采样生成回复
            # ============================================
            batch_prompts = random.sample(
                all_prompts, min(config["batch_size"], len(all_prompts)))
            batch = rollout(model=model, tokenizer=tokenizer, prompts_list=batch_prompts,
                            G=config["G"], max_new_tokens=config["max_new_tokens"],
                            temperature=config["temperature"])
            B, G = len(batch["prompts_text"]), config["G"]

            # ============================================
            # Phase 2: Reward & Advantage — 计算奖励与组内标准化
            # ============================================
            batch = compute_rewards_and_advantages(batch)
            rewards = batch["rewards"]          # [B*G]
            advantages = batch["advantages"]    # [B*G]
            reward_history.append(rewards.mean().item())

            # ---- 构建本步的 metrics 字典，统一收集所有指标 ----
            metrics = {}

            # 1) Reward / Advantage 指标
            metrics["reward/mean"]      = rewards.mean().item()
            metrics["reward/std"]       = rewards.std().item()
            metrics["reward/max"]       = rewards.max().item()
            metrics["reward/min"]       = rewards.min().item()

            metrics["advantage/mean"]      = advantages.mean().item()
            metrics["advantage/std"]       = advantages.std().item()
            metrics["advantage/max"]       = advantages.max().item()
            metrics["advantage/min"]       = advantages.min().item()

            # 2) 生成长度统计（检测模式崩溃或生成长度爆炸）
            response_ids = batch.get("response_ids")
            if response_ids is not None:
                if isinstance(response_ids, list):
                    lengths = [len(r) for r in response_ids]
                else:
                    lengths = (response_ids != tokenizer.pad_token_id).sum(dim=1).tolist()
            else:
                lengths = batch["response_mask"].sum(dim=1).int().tolist()
            metrics["generation/response_length_mean"] = sum(lengths) / len(lengths)
            metrics["generation/response_length_max"]  = max(lengths)
            metrics["generation/response_length_min"]  = min(lengths)
            metrics["generation/response_length_hist"] = swanlab.Histogram(lengths)
            metrics["generation/num_samples"]          = B * G

            # ============================================
            # Phase 3+4: 多轮 PPO 更新 (内层循环)
            # ============================================
            # 一次 rollout 后，对同一批数据做多次梯度更新（提高样本效率）
            for ppo_step in range(config["ppo_epochs"]):
                optimizer.zero_grad()
                # 重新计算 log-probs / loss（因为模型权重已更新）
                batch = compute_loss(model=model, ref_model=ref_model, batch=batch,
                                     eps_clip=config["eps_clip"], beta=config["beta"])

                loss = batch["loss"]
                kl_per_token = batch["kl_per_token"]        # [B*G, L_response]
                mask = batch["response_mask"].bool()
                valid_kl = (kl_per_token * mask).sum() / (mask.sum() + 1e-8)

                loss_history.append(loss.item())
                kl_history.append(valid_kl.item())

                # 计算 clip fraction —— PPO clip 机制被触发的 token 比例
                ratio = batch.get("ratio")
                clip_fraction = 0.0
                if ratio is not None:
                    clip_low  = 1.0 - config["eps_clip"]
                    clip_high = 1.0 + config["eps_clip"]
                    valid_ratio = ratio[mask]
                    clipped = ((valid_ratio < clip_low) | (valid_ratio > clip_high)).float()
                    clip_fraction = clipped.mean().item()

                # 用最后一次更新的值覆盖 metrics 中的训练指标
                metrics["loss/total"]            = loss.item()
                metrics["kl/mean"]               = valid_kl.item()
                metrics["kl/max"]                = (kl_per_token * mask).max().item() / 1.0
                metrics["policy/clip_fraction"]  = clip_fraction
                if ratio is not None:
                    metrics["policy/ratio_mean"]      = ratio[mask].mean().item()
                    metrics["policy/ratio_std"]       = ratio[mask].std().item()

                # 反向传播 + 梯度裁剪 + 优化器更新
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           config["grad_clip_norm"])
                optimizer.step()

                # 记录当前学习率与梯度范数
                metrics["train/lr"]        = scheduler.get_last_lr()[0]
                metrics["train/grad_norm"] = grad_norm

                # 若 KL 散度超出阈值，提前停止本轮更新（防止策略漂移过大）
                if valid_kl > config["target_kl"]:
                    print(f"  -> Early stop at ppo_step {ppo_step+1}, KL={valid_kl:.4f}")
                    break

            # ---- 一次性将所有指标写入 SwanLab ----
            metrics = {k: v for k, v in metrics.items() if v is not None}
            swanlab.log(metrics)

            # ---- 终端日志 / 保存 / 缓存清理 ----
            if global_step % config["log_interval"] == 0:
                log_metrics(global_step, {
                    "step":       global_step,
                    "epoch":      epoch + 1,
                    "lr":         metrics.get("train/lr", 0),
                    "loss":       metrics.get("loss/total", 0),
                    "reward_mean": metrics.get("reward/mean", 0),
                    "adv_mean":   metrics.get("advantage/mean", 0),
                    "mean_kl":    metrics.get("kl/mean", 0),
                    "grad_norm":  metrics.get("train/grad_norm", 0),
                    "num_samples": B * G,
                })

            if global_step % config["save_interval"] == 0:
                save_checkpoint(model, optimizer, scheduler, global_step,
                                {"loss": metrics.get("loss/total", 0.0),
                                 "reward_mean": metrics.get("reward/mean", 0.0),
                                 "mean_kl":    metrics.get("kl/mean", 0.0),
                                 "grad_norm":  metrics.get("train/grad_norm", 0.0)},
                                config)

            clear_cuda_cache(global_step)

        # 每个 epoch 结束：学习率按 gamma 衰减一次
        scheduler.step()

    # ============================================
    # 训练结束
    # ============================================
    print(f"\n{'#'*60}\n#  Training Complete!\n"
          f"#  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
          f"#  Total rollouts: {global_step}\n{'#'*60}")

    # 打印最终汇总
    print("\n[Summary]")
    if loss_history:
        print(f"  Loss:  min={min(loss_history):.4f}, max={max(loss_history):.4f}, final={loss_history[-1]:.4f}")
    if reward_history:
        print(f"  Reward: min={min(reward_history):.4f}, max={max(reward_history):.4f}, final={reward_history[-1]:.4f}")
    if kl_history:
        print(f"  KL:     min={min(kl_history):.4f}, max={max(kl_history):.4f}, final={kl_history[-1]:.4f}")

    # 保存最终模型
    final_metrics = {"loss": loss_history[-1] if loss_history else 0.0,
                     "reward_mean": reward_history[-1] if reward_history else 0.0,
                     "mean_kl": kl_history[-1] if kl_history else 0.0}
    save_checkpoint(model, optimizer, scheduler, global_step, final_metrics, config)
    print("\n[main] Done!")


if __name__ == "__main__":
    main()