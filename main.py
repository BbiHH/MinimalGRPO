"""
main.py — GRPO 完整训练流程 (性能优化版)
=====================================================
- 四个阶段：Rollout → Reward → Reference 缓存 → Loss 计算与梯度更新
- 优化点：显存泄漏修复、余弦退火调度器、Reference 前向传播冗余剔除
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
from loss import compute_loss, compute_ref_log_probs


CONFIG = {
    # --- 模型 ---
    "model_name":         "Qwen/Qwen2.5-0.5B-Instruct",

    # --- Rollout ---
    "prompt_file":        "data/prompts.jsonl",
    "batch_size":         4,              
    "G":                  8,              
    "max_new_tokens":     300,            
    "temperature":        0.8,            

    # --- Loss ---
    "eps_clip":           0.2,            
    "beta":               0.04,           
    "ppo_epochs":         4,              
    "target_kl":          0.02,           

    # --- 训练步数 ---
    "num_epochs":         10,             
    "steps_per_epoch":    50,             

    # --- 优化器 ---
    "learning_rate":      5e-6,
    "grad_clip_norm":     1.0,           

    # --- 日志与保存 ---
    "log_interval":       1,              
    "save_dir":           "checkpoints",
    "save_interval":      100,            

    # --- 资源 ---
    "dtype":              torch.bfloat16,
}


def setup_model_and_tokenizer(config):
    """加载 Actor 和 Reference，并严格对齐 pad_token_id 配置"""
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[main] Loading Actor & Reference models...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=config["dtype"], device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=config["dtype"], device_map="auto")
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Reference 模型冻结所有参数
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] Total params: {total/1e6:.1f}M, Actor trainable: {trainable/1e6:.1f}M")
    return model, ref_model, tokenizer


def setup_optimizer(model, config, total_steps):
    """AdamW 优化器 + 按全局步数平滑衰减的余弦调度器"""
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"],
                      betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )
    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, step, metrics, config):
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    # 保存完整状态字典用于断点续训
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }, os.path.join(save_dir, f"checkpoint_step_{step}.pt"))
    # 额外保存 HF 格式权重以便推理直接加载
    model.save_pretrained(os.path.join(save_dir, f"model_step_{step}"))


def log_metrics(step, metrics):
    """终端简洁日志打印"""
    print(f"\n[Step {step}] Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")


def clear_cuda_cache(step):
    """定期清理显存碎片，防患 OOM"""
    if step % 5 == 0:
        torch.cuda.empty_cache()


def main():
    config = CONFIG
    swanlab.init(project="MinimalGRPO", experiment_name="raw_2", config=config, mode="cloud")

    # 一次性加载所有 prompt
    all_prompts = []
    with open(config["prompt_file"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_prompts.append(json.loads(line)["prompt"])
    
    total_rollouts = config["num_epochs"] * config["steps_per_epoch"]
    print(f"[main] Loaded {len(all_prompts)} prompts. Total rollouts: {total_rollouts}")

    model, ref_model, tokenizer = setup_model_and_tokenizer(config)
    optimizer, scheduler = setup_optimizer(model, config, total_rollouts)

    loss_history, reward_history, kl_history = [], [], []
    global_step = 0

    # ==============================
    # 训练主循环
    # ==============================
    for epoch in range(config["num_epochs"]):
        print(f"\n{'='*40}\n EPOCH {epoch+1}/{config['num_epochs']} \n{'='*40}")

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

            # ============================================
            # Phase 2: Reward & Advantage 
            # ============================================
            batch = compute_rewards_and_advantages(batch)
            rewards = batch["rewards"]          
            advantages = batch["advantages"]    
            reward_history.append(rewards.mean().item())

            # 收集奖励与长度指标
            metrics = {}
            metrics["reward/mean"]      = rewards.mean().item()
            metrics["reward/std"]       = rewards.std().item()
            metrics["advantage/mean"]   = advantages.mean().item()
            metrics["advantage/std"]    = advantages.std().item()

            lengths = batch["response_mask"].sum(dim=1).int().tolist()
            metrics["generation/response_length_mean"] = sum(lengths) / len(lengths)
            metrics["generation/response_length_max"]  = max(lengths)
            metrics["generation/response_length_min"]  = min(lengths)

            # ============================================
            # Phase 3: Pre-compute Reference Log-Probs
            # ============================================
            # 仅执行一次 Reference 前向传播，将静态概率存入 batch
            batch["logp_ref"] = compute_ref_log_probs(ref_model, batch)

            # ============================================
            # Phase 4: 多轮 PPO 更新 (内层循环)
            # ============================================
            for ppo_step in range(config["ppo_epochs"]):
                optimizer.zero_grad()
                
                # compute_loss 不再需要 ref_model，极大提升内层循环速度
                batch = compute_loss(model=model, batch=batch,
                                     eps_clip=config["eps_clip"], beta=config["beta"])

                loss = batch["loss"]
                kl_per_token = batch["kl_per_token"]
                mask = batch["response_mask"].bool()
                valid_kl = (kl_per_token * mask).sum() / (mask.sum() + 1e-8)

                loss_history.append(loss.item())
                kl_history.append(valid_kl.item())

                # 计算被 clip 截断的 token 比例
                ratio = batch.get("ratio")
                clip_fraction = 0.0
                if ratio is not None:
                    clip_low  = 1.0 - config["eps_clip"]
                    clip_high = 1.0 + config["eps_clip"]
                    valid_ratio = ratio[mask]
                    clipped = ((valid_ratio < clip_low) | (valid_ratio > clip_high)).float()
                    clip_fraction = clipped.mean().item()

                metrics["loss/total"]            = loss.item()
                metrics["kl/mean"]               = valid_kl.item()
                metrics["policy/clip_fraction"]  = clip_fraction

                # 反向传播与梯度裁剪
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                optimizer.step()

                # 强制解包 Tensor，防止记录字典绑架计算图
                metrics["train/lr"]        = scheduler.get_last_lr()[0]
                metrics["train/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                # KL 散度超限预警与早停
                if valid_kl > config["target_kl"]:
                    print(f"  [Early Stop] ppo_step {ppo_step+1}, KL={valid_kl:.4f} > target({config['target_kl']})")
                    break

            # 学习率步进衰减
            scheduler.step()

            # ---- 写入 SwanLab 与 终端控制台 ----
            metrics = {k: v for k, v in metrics.items() if v is not None}
            swanlab.log(metrics)

            if global_step % config["log_interval"] == 0:
                log_metrics(global_step, {
                    "lr":         metrics.get("train/lr", 0),
                    "loss":       metrics.get("loss/total", 0),
                    "reward_mean": metrics.get("reward/mean", 0),
                    "mean_kl":    metrics.get("kl/mean", 0),
                })

            if global_step % config["save_interval"] == 0:
                save_checkpoint(model, optimizer, scheduler, global_step,
                                {"loss": metrics.get("loss/total", 0.0),
                                 "reward_mean": metrics.get("reward/mean", 0.0)}, config)

            clear_cuda_cache(global_step)

    # ============================================
    # 训练结束
    # ============================================
    print(f"\n[Training Complete] Total rollouts: {global_step}")
    final_metrics = {"loss": loss_history[-1] if loss_history else 0.0,
                     "reward_mean": reward_history[-1] if reward_history else 0.0}
    save_checkpoint(model, optimizer, scheduler, global_step, final_metrics, config)

if __name__ == "__main__":
    main()