"""
src/loss.py — Phase 3: Log-Probs、KL 散度与 GRPO 策略损失
============================================================

本模块负责：
  1. 对模型做前向传播，提取 response 部分每个 token 的对数概率 logist
  2. 计算 Actor 与 Reference 之间的 Token 级 KL 散度
  3. 构建基于 PPO-clip 的 GRPO 策略损失函数

核心数学（来自 README）：

  KL 近似：
      D_KL ≈ π_ref / π_θ - log(π_ref / π_θ) - 1

  重要性采样比率：
      ρ_i = exp(log π_θ(i) - log π_ref(i))

  GRPO 策略损失：
      L = -1/G * Σ_i [ min(ρ_i * A_i, clip(ρ_i, 1-ε, 1+ε) * A_i) - β * D_KL ]

输入：
  rollout_batch Dict（来自 rollout.py + reward.py），包含：
    - "full_ids"        — [B*G, L_total]    完整序列 token ids
    - "full_mask"       — [B*G, L_total]    完整 attention mask
    - "response_ids"    — [B*G, L_response] 纯 response token ids
    - "response_mask"   — [B*G, L_response] response 有效 token mask
    - "prompt_len"      — int, prompt 统一长度
    - "advantages"      — [B*G]  组内标准化优势值
    - "G"               — int, 组大小

输出：
  在 rollout_batch Dict 的基础上新增：
    - "loss"            — Tensor 标量，最终的 GRPO 损失值
    - "logp_actor"      — [B*G, L_response] Actor 模型每 token 的 log-prob
    - "logp_ref"        — [B*G, L_response] Reference 模型每 token 的 log-prob
    - "kl_per_token"    — [B*G, L_response] Token 级 KL 散度
    - "ratio"           — [B*G, L_response] 重要性比率 ρ
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

    核心逻辑：
        1. 将完整序列 full_ids 送入模型前向传播，得到 logits
        2. 对 logits 做 log_softmax，得到每个位置在所有词表上的 log-prob
        3. 使用 gather 操作，取出实际生成的 token 对应的 log-prob
        4. 截取 response 部分返回

    为什么是 response 部分而不是完整序列？
        GRPO 只需要优化「模型生成的 token」对应的概率。
        Prompt 部分是固定的，不应计入损失。

    为什么用 gather 而不是直接用 cross_entropy？
        - cross_entropy 直接给出标量平均损失，不便于后续 token 级加权
        - 用 gather 保留每个 token 的独立 log-prob，方便计算 ratio 和 clip

    为什么 shift by 1？
        - Causal LM 的 logits[i] 是在已知 token[0..i] 的情况下预测 token[i+1]
        - 所以取出 logits[i] 对应的是目标 token_ids[i+1]

    输入：
        model:      HuggingFace CausalLM（Actor 或 Reference）
        full_ids:   Tensor[B*G, L_total] — 完整序列 token ids
        full_mask:  Tensor[B*G, L_total] — 完整 attention mask
        prompt_len: int — prompt 统一序列长度

    输出：
        response_logp: Tensor[B*G, L_response]
                       每个 response token 在对应分布下的 log-prob

    实现细节与形状跟踪：

        full_ids:   [N, L_total]    其中 N = B*G
        ↓ model(...)
        logits:     [N, L_total, V]  V = vocab_size
        ↓ [:, :-1, :]  去掉最后一个位置（因为没有下一 token 预测目标）
        logits:     [N, L_total-1, V]
        ↓ log_softmax(dim=-1)
        log_probs:  [N, L_total-1, V]
        ↓ gather at target token ids
        targets = full_ids[:, 1:]     # 向右移一位
        对每个样本 n, 每个位置 t:
          logp[n,t] = log_probs[n, t, targets[n,t]]
        logp:       [N, L_total-1]
        ↓ 截取 response 部分 [:, prompt_len-1:]
        response_logp: [N, L_response]
    """
    # --- Step 1: 前向传播 ---
    # 获取模型所在设备
    device = next(model.parameters()).device
    full_ids = full_ids.to(device)
    full_mask = full_mask.to(device)

    # forward pass，获取每个位置的 raw logits
    outputs = model(
        input_ids=full_ids,
        attention_mask=full_mask,
    )
    logits = outputs.logits                     # [N, L_total, V]

    # --- Step 2: 对齐预测与目标 ---
    # Causal LM: logits[t] 预测 token[t+1]
    # 所以去掉最后一个 logits，向右 shift 1 位
    shift_logits = logits[:, :-1, :]             # [N, L_total-1, V]
    shift_targets = full_ids[:, 1:]              # [N, L_total-1]

    # --- Step 3: log_softmax + gather ---
    # log_softmax 把 logits 转成 log-probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [N, L_total-1, V]

    # gather 取出实际 token 对应的 log-prob
    # gather 的用法：log_probs[n, t, vocab_id] 中选 target_id
    # .unsqueeze(-1) 把 targets 从 [N, L-1] 变成 [N, L-1, 1]
    # gather 的结果是 [N, L-1, 1]，然后 squeeze(-1) 回到 [N, L-1]
    per_token_logp = log_probs.gather(
        dim=-1,
        index=shift_targets.unsqueeze(-1)
    ).squeeze(-1)                                # [N, L_total-1]

    # --- Step 4: 截取 response 部分 ---
    # prompt 有 prompt_len 个 token
    # shift 后位置对齐：log_probs[i] 对应 token[i+1]
    # 所以 token[prompt_len] 的 log-prob 位于 per_token_logp[:, prompt_len-1]
    response_logp = per_token_logp[:, prompt_len - 1:]  # [N, L_response]

    return response_logp


# ============================================================
# 第 2 部分：计算重要性比率与 KL 散度
# ============================================================

def compute_ratio_and_kl(
    logp_actor: torch.Tensor,
    logp_ref: torch.Tensor,
    response_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    功能：计算 Actor 与 Reference 之间的重要性比率 ρ 和 KL 散度

    重要性采样比率（Token 级）：
        ρ_i = exp(log π_θ(t_i | x, t_<i) - log π_ref(t_i | x, t_<i))
            = exp(logp_actor - logp_ref)

    这个比率衡量「当前策略对该 token 的采样概率」相对于「参考策略」的变化。
      ρ > 1：当前策略比参考策略更喜欢选这个 token
      ρ < 1：当前策略没参考策略那么喜欢这个 token

    KL 散度近似：
        D_KL ≈ π_ref/π_θ - log(π_ref/π_θ) - 1
             = exp(logp_ref - logp_actor) - (logp_ref - logp_actor) - 1
             = 1/ρ - (-log ρ) - 1
             = 1/ρ + log ρ - 1

    为什么用这个近似而不是标准 KL？
        标准 KL(π_ref || π_θ) = Σ π_ref * log(π_ref / π_θ)
        但这里我们用的是实际采样的 token，所以用单样本近似：
        D_KL_single ≈ p_ref/p_θ - log(p_ref/p_θ) - 1

        这被称为 f-divergence 中 f(x)=x-log(x)-1 的形式，
        在 TRL/DeepSpeed-Chat 等 RLHF 库中广泛使用。

    输入：
        logp_actor:     Tensor[N, L_response] — Actor 每 token 的 log-prob
        logp_ref:       Tensor[N, L_response] — Reference 每 token 的 log-prob
        response_mask:  Tensor[N, L_response] — 1=有效token, 0=padding

    输出：
        ratio:          Tensor[N, L_response] — 重要性比率 ρ
        kl_per_token:   Tensor[N, L_response] — Token 级 KL 散度
        log_ratio:      Tensor[N, L_response] — log ρ = logp_actor - logp_ref
    """
    # --- Step 0: 确保 mask 在同一设备 ---
    response_mask = response_mask.to(logp_actor.device)

    # --- Step 1: 计算 log-ratio ---
    log_ratio = logp_actor - logp_ref              # [N, L_response]

    # --- Step 2: 计算比率 ρ = exp(log_ratio) ---
    ratio = torch.exp(log_ratio)                   # [N, L_response]

    # --- Step 3: 计算 KL 散度 ---
    # D_KL ≈ exp(-log_ratio) + log_ratio - 1
    #       = 1/ratio + log_ratio - 1
    # 注意：当 ratio ≈ 1 (即两策略接近)时，KL ≈ 0
    kl_per_token = torch.exp(-log_ratio) + log_ratio - 1.0   # [N, L_response]

    # --- 数值稳定性检查 ---
    # KL 散度理论上应该 ≥ 0
    # 但浮点误差可能导致微小的负数
    kl_per_token = torch.clamp(kl_per_token, min=0.0)

    # 打印统计信息
    mask = response_mask.bool()
    valid_count = mask.sum().item()
    if valid_count > 0:
        mean_kl = (kl_per_token * response_mask).sum().item() / valid_count
        mean_ratio = (ratio * response_mask).sum().item() / valid_count
        print(f"[loss]   Mean ratio (masked): {mean_ratio:.4f}")
        print(f"[loss]   Mean KL (masked):    {mean_kl:.6f}")

    return ratio, kl_per_token, log_ratio


# ============================================================
# 第 3 部分：GRPO 策略损失函数
# ============================================================

def compute_grpo_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    kl_per_token: torch.Tensor,
    eps_clip: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """
    功能：计算 GRPO 策略损失（PPO-clip 变体 + KL 惩罚）

    数学公式：

        对每个 prompt 的 G 个 response 组，损失为：

            L = -1/G * Σ_i [ min(ρ_i * A_i, clip(ρ_i, 1-ε, 1+ε) * A_i) - β * D_KL(i) ]

        其中：
            ρ_i  — Token 级重要性比率
            A_i  — Token 级优势值（已在组内标准化，且扩展到每个 token）
            ε    — 裁剪范围，默认 0.2
            β    — KL 惩罚系数，默认 0.04

    为什么需要 clip？
        - PPO 的核心：防止策略更新过大
        - 当 ρ 偏离 1 太远时，clip 限制了梯度
        - 如果 A>0（好token），ρ 超过了 1+ε 就不再增大它的贡献（防止过度乐观更新）
        - 如果 A<0（坏token），ρ 低于了 1-ε 就不再减小它的贡献（防止过度悲观更新）

    为什么要加权取负平均？
        - 强化学习约定：最小化损失 → 最大化期望奖励
        - 因为 A_i 越高我们希望概率越大，所以 loss 是负的 ρ*A

    为什么是 token 级而不是 response 级？
        - 需要对每个生成的 token 分别优化
        - A_i 在组内是 response 级的（当前实现），但可以扩展到 token 级
        - 当前版本：同一个 response 的所有 token 共享同一个 advantage 值

    输入：
        ratio:          Tensor[N, L_response] — 重要性比率 ρ
        advantages:     Tensor[N]            — 每个样本的 advantage（response 级）
        response_mask:  Tensor[N, L_response] — response 有效 mask
        kl_per_token:   Tensor[N, L_response] — Token 级 KL 散度
        eps_clip:       float — PPO 裁剪范围
        beta:           float — KL 惩罚系数

    输出：
        loss:           Tensor 标量 — 最终损失值
    """
    # --- Step 1: 将 advantage 扩展到 token 级 ---
    # advantages: [N] → [N, 1] → [N, L_response]
    # 同一 response 的所有 token 共享相同 advantage
    # 确保 advantages 与 ratio 在同一设备上（避免 GPU/CPU 跨设备错误）
    advantages = advantages.to(ratio.device)
    adv_tokens = advantages.unsqueeze(-1).expand_as(ratio)   # [N, L_response]

    # --- Step 2: 计算未裁剪的梯度项 ---
    # 这就是 ρ_i * A_i 项
    pg_loss_unclipped = ratio * adv_tokens                  # [N, L_response]

    # --- Step 3: 计算裁剪后的梯度项 ---
    # clip(ρ, 1-ε, 1+ε) * A
    ratio_clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_loss_clipped = ratio_clipped * adv_tokens            # [N, L_response]

    # --- Step 4: 保守策略 — 取 min ---
    # 核心 PPO trick：A>0 时用 min 限制过度更新，A<0 时 min 也限制了惩罚
    # 这里的逻辑：
    #   若 A>0:  min(ρ*A, clip(ρ)*A) — 当 ρ 太大时限制贡献
    #   若 A<0:  min(ρ*A, clip(ρ)*A) — 当 ρ 太小时限制惩罚（即不做过多压制）
    #   因为 A<0 时剪裁版更小（更正），所以取 min 是对的
    pg_loss = torch.min(pg_loss_unclipped, pg_loss_clipped)  # [N, L_response]

    # --- Step 5: 减去 KL 惩罚 ---
    # KL 惩罚鼓励策略不要偏离参考策略太远
    loss_per_token = pg_loss - beta * kl_per_token          # [N, L_response]

    # --- Step 6: 对有效 token 求平均 ---
    # 只考虑 response mask 为 1 的 token
    # 然后除以 G（注意这里不是除以总 token 数，而是按组取平均）
    # 实际上对有效 token 求平均即可，因为最终目标是标量
    masked_loss = loss_per_token * response_mask             # [N, L_response]

    valid_lengths = response_mask.sum(dim=-1)                # [N] 每个样本的实际 token 数

    # 避免除以 0，使用 clamp
    valid_lengths = torch.clamp(valid_lengths, min=1.0)

    # 按样本聚合，每个 sample 的损失 ， 按样本取有效 token 的均值，而非直接 sum
    sample_losses = masked_loss.sum(dim=-1) / valid_lengths                  # [N]

    # 对 batch 中所有样本取平均
    total_loss = -sample_losses.mean()                       # 标量

    # 注意负号：因为我们是梯度上升期望奖励
    # RL 框架中通常写为 -E[ρ*A]，所以 loss = -(clipped_term - β*KL)

    return total_loss


# ============================================================
# 第 4 部分：主入口函数 — 串联全部计算
# ============================================================

def compute_loss(
    model,              # Actor 模型（可训练）
    ref_model,          # Reference 模型（冻结）
    batch: Dict[str, torch.Tensor],
    eps_clip: float = 0.2,
    beta: float = 0.04,
) -> Dict[str, torch.Tensor]:
    """
    功能：GRPO 损失计算主入口，一次调用完成全部 Phase 3 计算

    这是外部调用的唯一入口。

    执行流程：
        1. 从 batch 中提取所需张量
        2. 分别用 Actor 和 Reference 模型前向传播
        3. 提取 response 部分的 log-prob
        4. 计算比率 ρ 和 KL 散度
        5. 计算 GRPO 策略损失
        6. 将所有结果写回 batch dict

    参数：
        model:      HuggingFace CausalLM — Actor 模型（可训练）
        ref_model:  HuggingFace CausalLM — Reference 模型（冻结权重）
        batch:      Dict，来自 rollout.py + reward.py 的输出
                    必须包含：
                        "full_ids", "full_mask", "prompt_len"
                        "advantages", "response_mask", "G"
        eps_clip:   float — PPO 裁剪范围，默认 0.2
        beta:       float — KL 惩罚系数，默认 0.04

    返回值：
        batch:  同一个 Dict，新增以下 key：
            "loss"           — Tensor 标量，最终 GRPO 损失
            "logp_actor"     — [N, L_response] Actor 的 log-prob
            "logp_ref"       — [N, L_response] Reference 的 log-prob
            "kl_per_token"   — [N, L_response] Token 级 KL
            "ratio"          — [N, L_response] 重要性比率
    """
    # --- 提取 batch 中的关键数据 ---
    full_ids      = batch["full_ids"]          # [N, L_total]
    full_mask     = batch["full_mask"]         # [N, L_total]
    prompt_len    = batch["prompt_len"]        # int
    advantages    = batch["advantages"]        # [N]
    response_mask = batch["response_mask"]     # [N, L_response]
    G             = batch["G"]                 # int

    print(f"\n[loss] ===== Phase 3: Computing GRPO Loss =====")
    print(f"[loss] Input shapes: full_ids={list(full_ids.shape)}, "
          f"prompt_len={prompt_len}, G={G}")

    # --- Step 1: 计算 Actor 的 log-probs ---
    print(f"[loss] Step 1/4: Computing Actor log-probs...")
    logp_actor = compute_log_probs(
        model, full_ids, full_mask, prompt_len
    )
    print(f"[loss]   logp_actor shape: {list(logp_actor.shape)}")

    # --- Step 2: 计算 Reference 的 log-probs ---
    # Reference 模型冻结，不计算梯度，节省显存
    print(f"[loss] Step 2/4: Computing Reference log-probs...")
    with torch.no_grad():
        logp_ref = compute_log_probs(
            ref_model, full_ids, full_mask, prompt_len
        )
    print(f"[loss]   logp_ref shape:   {list(logp_ref.shape)}")

    # --- Step 3: 计算比率和 KL 散度 ---
    print(f"[loss] Step 3/4: Computing ratio and KL divergence...")
    ratio, kl_per_token, log_ratio = compute_ratio_and_kl(
        logp_actor, logp_ref, response_mask
    )
    print(f"[loss]   ratio shape:        {list(ratio.shape)}")
    new_shape = kl_per_token.shape
    print(f"[loss]   kl_per_token shape: {list(kl_per_token.shape)}")

    # --- Step 4: 计算 GRPO 损失 ---
    print(f"[loss] Step 4/4: Computing GRPO loss...")
    loss = compute_grpo_loss(
        ratio, advantages, response_mask, kl_per_token,
        eps_clip=eps_clip, beta=beta
    )
    print(f"[loss]   loss value: {loss.item():.6f}")

    # --- 写入 batch dict ---
    batch["loss"]         = loss
    batch["logp_actor"]   = logp_actor
    batch["logp_ref"]     = logp_ref
    batch["kl_per_token"]  = kl_per_token
    batch["ratio"]        = ratio

    print(f"[loss] ===== Phase 3 complete! =====")
    print(f"[loss] Added keys: loss, logp_actor, logp_ref, kl_per_token, ratio")

    return batch


@torch.no_grad()
def compute_ref_log_probs(
    ref_model,
    batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    功能：在 PPO 循环前预计算 Reference 模型的静态 log-probs。
    """
    full_ids = batch["full_ids"]
    full_mask = batch["full_mask"]
    prompt_len = batch["prompt_len"]
    
    logp_ref = compute_log_probs(ref_model, full_ids, full_mask, prompt_len)
    return logp_ref.detach() # 彻底切断与计算图的联系