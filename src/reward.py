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
import re
from tool import extract_calc_expressions, execute_calcs


# ============================================================
# 第 1 部分：奖励函数
# ============================================================

def compute_rewards(
    prompts_text: List[str],
    responses_text: List[str],
    G: int,
    answers: List = None,
    task_types: List[str] = None,
) -> torch.Tensor:
    """
    功能：为每个 (prompt, response) 计算标量奖励分数

    输入：
        prompts_text:   List[str], 长度 = B
                        （注意不是 B*G！每个 prompt 只有一条）
        responses_text: List[str], 长度 = B*G
                        （每个 prompt 有 G 个 response）
        G:              int，组大小
        answers:        List, 长度 = B*G（含 None）
                        标准答案，用于 tool-use 任务打分
        task_types:     List[str], 长度 = B*G（含 None）
                        任务类型，用于 reward 分发；默认全部为 "general"

    输出：
        rewards: Tensor[B*G]，dtype=float32
    """
    B = len(prompts_text)
    rewards_list = []

    for i, response in enumerate(responses_text):
        prompt_idx = i // G  # 这个 response 属于第几个 prompt
        prompt = prompts_text[prompt_idx]
        answer = answers[i] if answers is not None else None
        task_type = task_types[i] if task_types is not None else "general"

        score = _single_reward(prompt, response, task_type=task_type, answer=answer)
        rewards_list.append(score)

    rewards = torch.tensor(rewards_list, dtype=torch.float32)

    # 打印统计信息，帮助监控训练
    mean_r = rewards.mean().item()
    std_r = rewards.std().item()
    print(f"[reward] Computed {len(rewards_list)} rewards: "
          f"mean={mean_r:.3f}, std={std_r:.3f}, "
          f"min={rewards.min().item():.3f}, max={rewards.max().item():.3f}")

    return rewards


def _extract_box_answer(text):
    """
    提取response中的答案
    """

    pattern = r'<box>(.*?)</box>'
    return re.findall(pattern, text)


def _tool_reward(response, answer):
    """
    calculator 任务打分。

    分数构成：
    格式分:  <calc> 正确闭合 +0.3, <box> 正确闭合 +0.2  → 满分 0.5
    结果分:  answer 正确 +1.0
    工具分:  answer 正确 且 用了 calc → +0.3
    满分: 1.8

    设计原则：
    - 用工具计算 → 必须 <calc> 闭合
    - 给出计算结果 → 必须 <box> 闭合
    - 格式分只看这两条硬规矩，和答案对错无关
    - 结果分和工具分才是核心权重
    """
    score = 0.0

    # ---- 格式：两条硬规矩 ----
    # 规矩 1: 用工具计算 → 必须 <calc> 闭合
    calc_exprs = extract_calc_expressions(response)
    if len(calc_exprs) > 0:
        score += 0.3

    # 规矩 2: 给出结果 → 必须 <box> 闭合
    box_answers = _extract_box_answer(response)
    if len(box_answers) > 0:
        score += 0.2

    # ---- 结果分 ----
    answer_correct = False
    if len(box_answers) > 0:
        try:
            if abs(float(box_answers[-1].strip()) - float(answer)) < 1e-6:
                answer_correct = True
        except (ValueError, TypeError):
            pass

    if answer_correct:
        score += 1.0

    # ---- 工具分 ----
    if answer_correct and len(calc_exprs) > 0:
        score += 0.3

    return score

def _heuristic_reward(response: str) -> float:
    """
    功能：对单个 response 使用启发式规则计算奖励（不依赖 prompt 或 answer）

    输入：
        response: str，模型生成的 response 文本

    输出：
        score: float，范围 [0.0, 1.0]

    评分维度：
        1. 长度分数 — 根据 response 单词数给分
        2. 格式分数 — 排除空回复和乱码
        3. 内容基础分 — 固定基础分
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

def _degeneracy_penalty(response: str) -> float:
    """
    通用退化检测，所有任务类型共用。

    检测三种垃圾输出：
    - 循环重复：同一短语反复出现
    - 乱码：不可读的 token 序列
    - 空回复：没生成任何有效内容
    """
    penalty = 0.0
    text = response.strip()
    words = text.split()
    n_words = len(words)

    # 1. 空回复或极短
    if n_words < 3:
        return -0.5

    # 2. 循环重复检测（3-gram 唯一比例）
    if n_words >= 9:
        trigrams = [tuple(words[i:i+3]) for i in range(n_words - 2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        if unique_ratio < 0.25:      # < 25% 唯一 → 严重重复
            penalty -= 0.5
        elif unique_ratio < 0.4:     # 25-40% → 轻度重复
            penalty -= 0.2

    # 3. 乱码检测
    # 平均词长 > 20 → 大概率是随机 token 拼成的
    avg_word_len = sum(len(w) for w in words) / n_words
    if avg_word_len > 20:
        penalty -= 0.5

    # 非字母非空白字符占比 > 70% → 符号乱码
    non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
    if len(text) > 10 and non_alpha / len(text) > 0.7:
        penalty -= 0.3

    return penalty

def _single_reward(prompt: str, response: str, task_type: str = "general", answer=None) -> float:
    if task_type == "calculator":
        score = _tool_reward(response, answer)
    else:
        score = _heuristic_reward(response)

    # 通用退化惩罚，所有任务类型生效
    score += _degeneracy_penalty(response)

    return score


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
    answers        = rollout_batch.get("answers", None) # List, len=B*G (向后兼容旧 batch)
    task_types     = rollout_batch.get("task_types", None)  # List[str], len=B*G (向后兼容旧 batch)
    G = rollout_batch["G"]

    print(f"[reward] Computing rewards for {len(responses_text)} responses...")

    # Step 1: 计算原始奖励
    rewards = compute_rewards(prompts_text, responses_text, G, answers, task_types)

    # Step 2: 组内标准化 → Advantage
    advantages = compute_advantages(rewards, G)

    # Step 3: 写入 batch dict
    rollout_batch["rewards"]    = rewards
    rollout_batch["advantages"] = advantages

    print(f"[reward] Done. rewards shape={rewards.shape}, "
          f"advantages shape={advantages.shape}")

    return rollout_batch