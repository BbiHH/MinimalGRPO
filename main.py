"""
main.py — GRPO 完整训练流程 (性能优化版)
=====================================================
- 四个阶段：Rollout → Reward → Reference 缓存 → Loss 计算与梯度更新
- 优化点：显存泄漏修复、余弦退火调度器、Reference 前向传播冗余剔除
"""

import os
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import swanlab
import sys
from typing import List

sys.path.insert(0, "src")
from rollout import rollout, PromptExample, TOOL_SYSTEM_PROMPT, load_prompts
from reward import compute_rewards_and_advantages, _extract_box_answer
from loss import compute_loss, compute_ref_log_probs
from tool import extract_calc_expressions


CONFIG = {
    # --- 模型 ---
    "model_name":         "Qwen/Qwen2.5-1.5B-Instruct",

    # --- Rollout ---
    "calculator_prompt_file":  "data/tool_calling_prompts.jsonl",
    "general_prompt_file":     "data/prompts.jsonl",    # None = calculator-only
    "use_mixed_training":      None,                     # True = 混合 calculator + general
    "batch_size":              2,
    "G":                       8,
    "max_new_tokens":          200,           # tool-use 输出更长，需更多 token
    "temperature":             0.8,

    # --- Loss ---
    "eps_clip":           0.2,
    "beta":               0.03,
    "ppo_epochs":         4,
    "target_kl":          0.04,

    # --- 训练步数 ---
    "num_epochs":         10,
    "steps_per_epoch":    100,

    # --- 优化器 ---
    "learning_rate":      1e-6,
    "grad_clip_norm":     1.0,

    # --- 日志与保存 ---
    "log_interval":       1,
    "save_dir":           "checkpoints",
    "save_interval":      500,

    # --- 资源与复现 ---
    "dtype":              torch.bfloat16,
    "seed":               42,
}


def load_training_data(calculator_file: str, general_file: str = None) -> List[PromptExample]:
    """
    加载并合并多个 prompt 数据集，构建统一训练池。

    Design decisions（设计决策）:
      - Calculator 示例注入 TOOL_SYSTEM_PROMPT（per-example），教模型使用工具
      - General 示例的 system_prompt 保持 None（无工具说明），保持通用对话能力
      - 混合比例 = 各数据集大小比例（自然平衡），不做手动加权
      - 返回扁平池供 random.sample() 每步随机采样

    职责边界：
      load_prompts() 负责文件 I/O → 返回"裸" PromptExample
      load_training_data() 负责训练配置 → 按数据集来源注入 system_prompt
      两者分工明确：一个读数据，一个配训练
    """
    pool: List[PromptExample] = []

    # 加载 calculator 示例（带工具 system prompt）
    calc_examples = load_prompts(calculator_file)
    for ex in calc_examples:
        ex.system_prompt = TOOL_SYSTEM_PROMPT   # 教模型何时使用 <calc> 标签
    pool.extend(calc_examples)
    print(f"[main] Loaded {len(calc_examples)} calculator examples "
          f"(system_prompt=TOOL_SYSTEM_PROMPT)")

    # 可选加载 general 示例（无 system prompt）
    if general_file is not None:
        gen_examples = load_prompts(general_file)
        # system_prompt 保持 None —— 通用问答不需要工具说明
        pool.extend(gen_examples)
        print(f"[main] Loaded {len(gen_examples)} general examples "
              f"(system_prompt=None)")

    # 打印训练池组成
    type_counts: Dict[str, int] = {}
    for ex in pool:
        type_counts[ex.task_type] = type_counts.get(ex.task_type, 0) + 1
    has_sys = sum(1 for ex in pool if ex.system_prompt is not None)
    print(f"[main] Training pool: {len(pool)} total, "
          f"composition={type_counts}, "
          f"with_system_prompt={has_sys}/{len(pool)}")

    return pool


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

    # --- 复现性：固定随机种子 ---
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # --- 构建训练池：混合 calculator + general ---
    # load_training_data 负责：
    #   1. 从 JSONL 加载原始数据（调用 load_prompts）
    #   2. 按数据集来源注入 per-example system_prompt
    #   3. 合并为扁平池供随机采样
    general_file = config["general_prompt_file"] if config["use_mixed_training"] else None
    training_pool = load_training_data(
        config["calculator_prompt_file"],
        general_file=general_file,
    )

    total_rollouts = config["num_epochs"] * config["steps_per_epoch"]
    print(f"[main] Training pool size: {len(training_pool)}. "
          f"Total rollouts: {total_rollouts}")

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
            # 每步从训练池随机采样 batch，以 PromptExample 列表传入
            batch_examples = random.sample(
                training_pool, min(config["batch_size"], len(training_pool)))
            batch = rollout(model=model, tokenizer=tokenizer, examples=batch_examples,
                            G=config["G"], max_new_tokens=config["max_new_tokens"],
                            temperature=config["temperature"])

            # ============================================
            # Phase 2: Reward & Advantage
            # ============================================
            batch = compute_rewards_and_advantages(batch)
            rewards = batch["rewards"]
            advantages = batch["advantages"]
            reward_history.append(rewards.mean().item())

            # 收集基础指标
            metrics = {}
            metrics["reward/mean"]      = rewards.mean().item()
            metrics["reward/std"]       = rewards.std().item()
            metrics["advantage/mean"]   = advantages.mean().item()
            metrics["advantage/std"]    = advantages.std().item()

            lengths = batch["response_mask"].sum(dim=1).int().tolist()
            metrics["generation/response_length_mean"] = sum(lengths) / len(lengths)
            metrics["generation/response_length_max"]  = max(lengths)
            metrics["generation/response_length_min"]  = min(lengths)

            # --- 组内奖励离散度：同一 prompt 的 G 个 response 之间 reward 的标准差 ---
            # 高 = 组内区分度大，GRPO 的 advantage 信号强
            # 低（< 0.1）= response 缺乏多样性，advantage 接近 0，GRPO 训练信号弱
            B = len(batch["prompts_text"])
            rewards_2d = rewards.view(B, config["G"])                 # [B, G]
            within_group_std = rewards_2d.std(dim=-1).mean().item()   # scalar
            metrics["generation/within_group_reward_std"] = within_group_std

            # ============================================
            # Phase 2.5: Tool-Use 监控指标
            # ============================================
            # 只在 batch 中包含 calculator 示例时才有意义
            calc_mask = [t == "calculator" for t in batch["task_types"]]
            if any(calc_mask):
                responses_text = batch["responses_text"]   # len = B*G
                answers_list = batch["answers"]            # len = B*G

                # 筛选 calculator 任务的 response 和 answer
                calc_responses = [responses_text[i] for i, m in enumerate(calc_mask) if m]
                calc_answers   = [answers_list[i] for i, m in enumerate(calc_mask) if m]

                # tool/use_rate: 至少包含一个 <calc> 标签的 response 比例
                has_calc_flags = [1 if len(extract_calc_expressions(r)) > 0 else 0
                                  for r in calc_responses]
                metrics["tool/use_rate"] = (sum(has_calc_flags) / len(has_calc_flags)
                                            if has_calc_flags else 0.0)

                # tool/answer_accuracy: <box> 答案与 ground truth 匹配的比例
                correct = 0
                for resp, ans in zip(calc_responses, calc_answers):
                    box_vals = _extract_box_answer(resp)
                    if box_vals and ans is not None:
                        try:
                            if abs(float(box_vals[-1].strip()) - float(ans)) < 1e-6:
                                correct += 1
                        except (ValueError, TypeError):
                            pass
                metrics["tool/answer_accuracy"] = (correct / len(calc_responses)
                                                    if calc_responses else 0.0)

                # tool/format_score_mean: calc+box 格式分的平均值
                format_scores = []
                for resp in calc_responses:
                    fs = 0.0
                    if len(extract_calc_expressions(resp)) > 0:
                        fs += 0.3
                    if len(_extract_box_answer(resp)) > 0:
                        fs += 0.2
                    format_scores.append(fs)
                metrics["tool/format_score_mean"] = (sum(format_scores) / len(format_scores)
                                                      if format_scores else 0.0)

                # reward/tool_mean: calculator 任务的平均奖励
                calc_rewards = [rewards[i].item() for i, m in enumerate(calc_mask) if m]
                metrics["reward/tool_mean"] = (sum(calc_rewards) / len(calc_rewards)
                                                if calc_rewards else 0.0)

                # reward/general_mean: general 任务的平均奖励
                gen_rewards = [rewards[i].item() for i, m in enumerate(calc_mask) if not m]
                metrics["reward/general_mean"] = (sum(gen_rewards) / len(gen_rewards)
                                                   if gen_rewards else 0.0)

            # --- 组内回复多样性：同一 prompt 的 G 个回复中，两两不同的比例 ---
            # 1.0 = 完全多样（GRPO 组内比较有意义）
            # < 0.3 = 回复趋于同质化，advantage 退化为噪声，需要增大 temperature 或调整 reward
            responses_text = batch["responses_text"]  # len = B*G
            G = config["G"]
            total_pairs = 0
            distinct_pairs = 0
            for b_idx in range(len(batch["prompts_text"])):
                group_responses = responses_text[b_idx * G : (b_idx + 1) * G]
                for i in range(G):
                    for j in range(i + 1, G):
                        total_pairs += 1
                        if group_responses[i].strip() != group_responses[j].strip():
                            distinct_pairs += 1
            metrics["generation/response_distinct_2"] = (
                distinct_pairs / total_pairs if total_pairs > 0 else 0.0
            )

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

                # 策略级监控指标（由 loss.py 写入 batch dict，已为 scalar float）
                if "entropy_mean" in batch:
                    metrics["policy/entropy_mean"] = batch["entropy_mean"]
                if "ratio_mean" in batch:
                    metrics["policy/ratio_mean"] = batch["ratio_mean"]
                if "ratio_max" in batch:
                    metrics["policy/ratio_max"] = batch["ratio_max"]

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
                    "lr":              metrics.get("train/lr", 0),
                    "loss":            metrics.get("loss/total", 0),
                    "reward_mean":     metrics.get("reward/mean", 0),
                    "mean_kl":         metrics.get("kl/mean", 0),
                    "entropy_mean":    metrics.get("policy/entropy_mean", 0),
                    "tool_use_rate":   metrics.get("tool/use_rate", 0),
                    "tool_answer_acc": metrics.get("tool/answer_accuracy", 0),
                    "within_group_std": metrics.get("generation/within_group_reward_std", 0),
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