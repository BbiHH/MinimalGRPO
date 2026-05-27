"""
src/reward.py — Phase 2: Reward & Advantage 计算模块
======================================================

本模块负责：
  1. 对每个 (prompt, response) 对计算标量奖励分数
  2. 在组内（同一 prompt 的 G 个 response）标准化为 Advantage

核心数学（来自 README） GRPO 优势计算公式 ：
  A_i = (R_i - μ组) / (σ组 + ε)
  
  其中 μ组 = mean(R_1...R_G), σ组 = std(R_1...R_G), ε = 1e-4

输入：
  rollout() 返回的 Dict，包含：
    - "responses_text": List[str], len = B*G
    - "prompts_text":   List[str], len = B
    - "G":              int

输出：
  在 rollout Dict 的基础上新增以下 key：
    - "rewards":        Tensor[B*G] — 每个样本的标量奖励
    - "advantages":     Tensor[B*G] — 组内标准化后的优势值
"""

import torch
from typing import Dict, List


# ============================================================
# 第 1 部分：奖励函数
# ============================================================

def compute_rewards(
    prompts_text: List[str],
    responses_text: List[str],
    G: int,
) -> torch.Tensor:
    """
    功能：为每个 (prompt, response) 计算标量奖励分数

    输入：
        prompts_text:   List[str], 长度 = B
                        （注意不是 B*G！每个 prompt 只有一条）
        responses_text: List[str], 长度 = B*G
                        （每个 prompt 有 G 个 response）
        G:              int，组大小

    输出：
        rewards: Tensor[B*G]，dtype=float32

    实现：
        当前使用规则奖励（rule-based reward），不引入额外模型。
        未来可替换为训练好的 reward model。

    奖励规则（针对不同任务可自定义）：
        规则 1 — 长度奖励：适当长度（30-100 token）得分高
        规则 2 — 格式奖励：不包含乱码/重复
        规则 3 — 关键词奖励：包含问题相关的关键词（可选）

    为什么用规则而不直接用"正确性"？
        因为大多数开放回答没有标准答案。我们先用通用的启发式规则，
        确保框架跑通。后续针对具体任务再设计精确奖励。
    """
    B = len(prompts_text)
    rewards_list = []

    for i, response in enumerate(responses_text):
        prompt_idx = i // G  # 这个 response 属于第几个 prompt
        prompt = prompts_text[prompt_idx]

        score = _single_reward(prompt, response)
        rewards_list.append(score)

    rewards = torch.tensor(rewards_list, dtype=torch.float32)

    # 打印统计信息，帮助监控训练
    mean_r = rewards.mean().item()
    std_r = rewards.std().item()
    print(f"[reward] Computed {len(rewards_list)} rewards: "
          f"mean={mean_r:.3f}, std={std_r:.3f}, "
          f"min={rewards.min().item():.3f}, max={rewards.max().item():.3f}")

    return rewards


def _single_reward(prompt: str, response: str) -> float:
    """
    功能：对单个 (prompt, response) 对计算奖励

    输入：
        prompt:   str，原始 prompt 文本
        response: str，模型生成的 response 文本

    输出：
        score: float，范围通常 [0.0, 1.0]

    规则详解：
        (1) 长度分 (0.0 - 0.5)
            太短（<10字）：可能没说完 → 低分
            太短（10-20字）：简单回答但可能不完整 → 中等
            合适（20-100字）：可能比较详细 → 高分
            太长（>100字）：可能啰嗦或跑题 → 中等偏低
            公式：len_score = 0.5 * min(1.0, (len-10) / 90)
                  确保 10→0, 100→0.5

        (2) 格式分 (0.0 - 0.3)
            如果 response 只包含空白/特殊字符 → 0
            如果包含正常可读内容 → 0.3

        (3) 内容分 (0.0 - 0.2)
            暂时给固定基础分，后续可接入更复杂的评估
    """
    response = response.strip()

    # --- 子规则 1：长度分数 ---
    # 简单 token 估算（英文按空格切分即可近似）
    word_count = len(response.split())

    if word_count < 10:
        length_score = 0.0          # 太短，没生成有效内容
    elif word_count < 45:
        length_score = 0.15         # 有基本内容但不够详细
    elif word_count <= 120:
        length_score = 0.4          # 理想长度范围
    elif word_count <= 180:
        length_score = 0.3          # 偏长但可接受
    else:
        length_score = 0.1          # 太长，可能跑题

    # --- 子规则 2：格式分数 ---
    # 排除空回复和纯乱码
    if len(response) == 0:
        format_score = 0.0
    elif _is_gibberish(response):
        format_score = 0.0
    else:
        format_score = 0.3

    # --- 子规则 3：内容基础分 ---
    content_score = 0.2

    # --- 合并总分 ---
    total = length_score + format_score + content_score

    # 确保在 [0, 1] 范围内
    total = max(0.0, min(1.0, total))

    return total


def _is_gibberish(text: str) -> bool:
    """
    功能：检测文本是否像乱码（用于格式惩罚）

    启发式规则：
        - 如果文本中单词的平均长度 > 15 字符，大概率是乱码
        - 如果 ASCII 标点占比异常高
    """
    words = text.split()
    if len(words) == 0:
        return True

    # 检查平均词长
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len > 20:
        return True

    # 检查是否大量重复同一个词
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.3 and len(words) > 10:
        return True

    return False


# ============================================================
# 第 2 部分：组内 Advantage 标准化
# ============================================================

def compute_advantages(
    rewards: torch.Tensor,
    G: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    功能：在组内标准化奖励，得到 Advantage

    输入：
        rewards: Tensor[B*G]         — 每个样本的原始奖励
        G:       int                 — 组大小
        eps:     float               — 防止除零的极小量

    输出：
        advantages: Tensor[B*G]      — 组内标准化后的优势值

    数学公式（对每个组）：
        A_i = (R_i - μ) / (σ + ε)

    张量形状变换逻辑（以 B=2, G=4 为例）：
        rewards:         [8]    = [R0, R0, R0, R0,  R1, R1, R1, R1]
          ↓ reshape
        rewards_2d:      [2, 4] = [[R0,R0,R0,R0],
                                   [R1,R1,R1,R1]]
          ↓ .mean(dim=-1)       → μ:   [2]    = [μ0, μ1]
          ↓ .std(dim=-1)        → σ:   [2]    = [σ0, σ1]
          ↓ .unsqueeze          → μ:   [2, 1]
          ↓ .expand             → μ:   [2, 4]
          ↓ .reshape(-1)        → μ:   [8]   ready for element-wise op

    为什么 (B*G) → (B, G) 这种 reshape 是正确的？
        因为 expand_for_group() 保证了数据布局：
        前 G 个样本属于组 0，下 G 个属于组 1，...
        所以直接 reshape 到 (B, G) 就是按组排列的。
    """
    # Step 1: 确定 B
    total_samples = rewards.shape[0]
    B = total_samples // G
    assert B * G == total_samples, \
        f"rewards 长度 {total_samples} 不能被 G={G} 整除！"

    # Step 2: reshape 成 (B, G)，每行是一个组
    rewards_2d = rewards.view(B, G)   # [B, G]

    # Step 3: 计算每组的均值和标准差
    mean_per_group = rewards_2d.mean(dim=-1)    # [B]
    std_per_group  = rewards_2d.std(dim=-1)     # [B]

    # Step 4: 扩展回 (B, G)，然后展平
    mean_expanded = mean_per_group.unsqueeze(-1).expand(-1, G)  # [B, G]
    std_expanded  = std_per_group.unsqueeze(-1).expand(-1, G)   # [B, G]

    mean_flat = mean_expanded.reshape(-1)   # [B*G]
    std_flat  = std_expanded.reshape(-1)    # [B*G]

    # Step 5: 计算标准化 Advantage
    advantages = (rewards - mean_flat) / (std_flat + eps)

    # 验证：组内 Advantage 均值应接近 0（理论上是精确 0）
    adv_2d = advantages.view(B, G)
    group_mean = adv_2d.mean(dim=-1)
    zero_check = group_mean.abs().max().item()
    print(f"[reward] Advantages computed. "
          f"Group mean check (should ≈ 0): max|μ_adv| = {zero_check:.2e}")

    return advantages


# ============================================================
# 第 3 部分：主入口函数
# ============================================================

def compute_rewards_and_advantages(
    rollout_batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    功能：对 rollout 的输出计算 reward 和 advantage，并添加到 dict 中

    输入：
        rollout_batch: rollout() 的输出 Dict
                       必须包含 "prompts_text", "responses_text", "G"

    输出：
        同一个 Dict，新增以下 key：
            "rewards":    Tensor[B*G]，标量奖励
            "advantages": Tensor[B*G]，组内标准化优势值

    该函数是外部调用的唯一入口。
    一次调用完成全部 reward + advantage 计算。
    """
    prompts_text   = rollout_batch["prompts_text"]     # List[str], len=B
    responses_text = rollout_batch["responses_text"]   # List[str], len=B*G
    G = rollout_batch["G"]

    print(f"[reward] Computing rewards for {len(responses_text)} responses...")

    # Step 1: 计算原始奖励
    rewards = compute_rewards(prompts_text, responses_text, G)

    # Step 2: 组内标准化 → Advantage
    advantages = compute_advantages(rewards, G)

    # Step 3: 写入 batch dict
    rollout_batch["rewards"]    = rewards
    rollout_batch["advantages"] = advantages

    print(f"[reward] Done. rewards shape={rewards.shape}, "
          f"advantages shape={advantages.shape}")

    return rollout_batch