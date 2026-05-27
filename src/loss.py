"""
src/loss.py — Phase 3: Log-Probs、KL 散度与 GRPO 策略损失
============================================================

本模块负责：
  1. 对模型做前向传播，提取 response 部分每个 token 的对数概率 logits
  2. 计算 Actor 与 Reference 之间的 Token 级 KL 散度
  3. 构建基于 PPO-clip 的 GRPO 策略损失函数

核心数学：
  KL 近似：D_KL ≈ π_ref / π_θ - log(π_ref / π_θ) - 1
  重要性采样比率：ρ_i = exp(log π_θ(i) - log π_ref(i))
  GRPO 策略损失：L = -1/G * Σ_i [ min(ρ_i * A_i, clip(ρ_i, 1-ε, 1+ε) * A_i) - β * D_KL ]
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


# ============================================================
# 第 1 部分：提取 Response Token 的 Log-Probabilities
# ============================================================
def compute_log_probs(
    model,
    full_ids: torch.Tensor,
    full_mask: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    功能：对模型做一次前向传播，提取 response 部分每个 token 的 log-prob
    """
    # --- Step 1: 前向传播 ---
    device = next(model.parameters()).device
    full_ids = full_ids.to(device)
    full_mask = full_mask.to(device)

    # 获取每个位置的 raw logits: [N, L_total, V]
    outputs = model(
        input_ids=full_ids,
        attention_mask=full_mask,
    )
    logits = outputs.logits

    # --- Step 2: 对齐预测与目标 ---
    # Causal LM: logits[t] 预测 token[t+1]，向右 shift 1 位
    shift_logits = logits[:, :-1, :]             # [N, L_total-1, V]
    shift_targets = full_ids[:, 1:]              # [N, L_total-1]

    # --- Step 3: log_softmax + gather ---
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [N, L_total-1, V]

    # 取出实际 token 对应的 log-prob
    per_token_logp = log_probs.gather(
        dim=-1,
        index=shift_targets.unsqueeze(-1)
    ).squeeze(-1)                                # [N, L_total-1]

    # --- Step 4: 截取 response 部分 ---
    response_logp = per_token_logp[:, prompt_len - 1:]  # [N, L_response]

    return response_logp


# ============================================================
# 新增：预计算 Reference Log-Probs (用于彻底剥离计算图)
# ============================================================
@torch.no_grad()
def compute_ref_log_probs(
    ref_model,
    batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    功能：在 PPO 循环前预计算 Reference 模型的静态 log-probs
    优化：避免在多轮 PPO epoch 中重复对冻结模型做前向传播
    """
    full_ids = batch["full_ids"]
    full_mask = batch["full_mask"]
    prompt_len = batch["prompt_len"]
    
    logp_ref = compute_log_probs(ref_model, full_ids, full_mask, prompt_len)
    
    # 彻底切断与计算图的联系，只保留纯张量数据
    return logp_ref.detach()


# ============================================================
# 第 2 部分：计算重要性比率与 KL 散度
# ============================================================
def compute_ratio_and_kl(
    logp_actor: torch.Tensor,
    logp_ref: torch.Tensor,
    response_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算重要性比率 ρ 和 单样本近似 KL 散度
    D_KL ≈ 1/ratio + log_ratio - 1
    """
    response_mask = response_mask.to(logp_actor.device)
    
    # 计算 log-ratio 与 比率 ρ
    log_ratio = logp_actor - logp_ref              # [N, L_response]
    ratio = torch.exp(log_ratio)                   # [N, L_response]

    # 计算 KL 散度
    kl_per_token = torch.exp(-log_ratio) + log_ratio - 1.0   # [N, L_response]
    kl_per_token = torch.clamp(kl_per_token, min=0.0)        # 浮点误差保护

    return ratio, kl_per_token, log_ratio


# ============================================================
# 第 3 部分：GRPO 策略损失函数 (已修复长度偏置)
# ============================================================
def compute_grpo_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    kl_per_token: torch.Tensor,
    eps_clip: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """计算 GRPO 策略损失，消除了序列长度偏置"""
    # 将 advantage 扩展到 token 级
    advantages = advantages.to(ratio.device)
    adv_tokens = advantages.unsqueeze(-1).expand_as(ratio)   # [N, L_response]

    # 未裁剪与裁剪的梯度项
    pg_loss_unclipped = ratio * adv_tokens                  # [N, L_response]
    ratio_clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_loss_clipped = ratio_clipped * adv_tokens            # [N, L_response]

    # 取 min 限制过度更新
    pg_loss = torch.min(pg_loss_unclipped, pg_loss_clipped)  # [N, L_response]
    
    # 减去 KL 惩罚
    loss_per_token = pg_loss - beta * kl_per_token          # [N, L_response]

    # 只考虑有效 token
    masked_loss = loss_per_token * response_mask             # [N, L_response]

    # --- 优化点：消除序列长度偏置 ---
    # 计算每个样本实际生成的有效 token 数量，防止长文本主导梯度
    valid_lengths = response_mask.sum(dim=-1)                # [N]
    valid_lengths = torch.clamp(valid_lengths, min=1.0)      # 防止除以 0

    # 按样本取均值，而非单纯 sum
    sample_losses = masked_loss.sum(dim=-1) / valid_lengths  # [N]

    # 最终取 batch 平均，加负号（梯度上升期望奖励）
    total_loss = -sample_losses.mean()                       # 标量

    return total_loss


# ============================================================
# 第 4 部分：主入口函数
# ============================================================
def compute_loss(
    model,              # 仅需要 Actor 模型
    batch: Dict[str, torch.Tensor],
    eps_clip: float = 0.2,
    beta: float = 0.04,
) -> Dict[str, torch.Tensor]:
    """GRPO 损失计算主入口，不再接受 ref_model，直接读取常量 logp_ref"""
    full_ids      = batch["full_ids"]          
    full_mask     = batch["full_mask"]         
    prompt_len    = batch["prompt_len"]        
    advantages    = batch["advantages"]        
    response_mask = batch["response_mask"]     
    
    # 直接读取缓存的常量 Reference log-probs
    if "logp_ref" not in batch:
        raise ValueError("[Error] batch dict must contain 'logp_ref'. "
                         "Please call compute_ref_log_probs before PPO epochs.")
    logp_ref = batch["logp_ref"]

    # Step 1: 仅对当前策略 Actor 做动态前向传播
    logp_actor = compute_log_probs(model, full_ids, full_mask, prompt_len)

    # Step 2: 计算比率和 KL 散度
    ratio, kl_per_token, log_ratio = compute_ratio_and_kl(
        logp_actor, logp_ref, response_mask
    )

    # Step 3: 计算无偏置 GRPO 损失
    loss = compute_grpo_loss(
        ratio, advantages, response_mask, kl_per_token,
        eps_clip=eps_clip, beta=beta
    )

    # 写入 batch
    batch["loss"]         = loss
    batch["logp_actor"]   = logp_actor
    batch["kl_per_token"] = kl_per_token
    batch["ratio"]        = ratio

    return batch