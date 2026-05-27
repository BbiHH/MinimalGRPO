"""
test_phase1_2.py — 验证 Rollout + Reward/Advantage 全链路
==========================================================
测试内容：
  1. 加载 Qwen2.5-0.5B 模型和 tokenizer
  2. 从 data/prompts.jsonl 加载 prompts，执行 rollout
  3. 计算 reward 和 group-advantege
  4. 打印每个组的结果，验证组内 standardize 是否正确
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入我们自己写的模块
import sys
sys.path.insert(0, "src")
from rollout import rollout, load_prompts, tokenize_prompts, expand_for_group
from reward import compute_rewards_and_advantages

# ============================================================
# Step 1: 加载模型和 tokenizer
# ============================================================
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("=" * 60)
print("[main] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 设置 pad_token（Qwen2.5 默认没有）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[main] pad_token_id = {tokenizer.pad_token_id}")
print(f"[main] eos_token_id = {tokenizer.eos_token_id}")

print("[main] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"[main] Model loaded on: {next(model.parameters()).device}")
print("=" * 60)


# ============================================================
# Step 2: 执行 Rollout
# ============================================================
print("\n[Phase 1] Starting rollout...")
print("-" * 60)

batch = rollout(
    model=model,
    tokenizer=tokenizer,
    prompt_file="data/prompts.jsonl",
    G=4,                    # 每个 prompt 生成 4 个 response
    max_new_tokens=300,      # 最多生成 50 token（测试用，快）
    temperature=0.5,
)
print("-" * 60)


# ============================================================
# Step 3: 计算 Reward + Advantage
# ============================================================
print("\n[Phase 2] Computing rewards and advantages...")
print("-" * 60)

batch = compute_rewards_and_advantages(batch)
print("-" * 60)


# ============================================================
# Step 4: 验证结果 — 按组展示
# ============================================================
print("\n" + "=" * 60)
print("[Result] Per-group breakdown")
print("=" * 60)

# 从 batch 中提取数据
B     = len(batch["prompts_text"])
G     = batch["G"]
rewards    = batch["rewards"]       # [B*G]
advantages = batch["advantages"]    # [B*G]
responses  = batch["responses_text"]

for p_idx in range(B):
    print(f"\n{'─' * 60}")
    print(f"📝 Prompt {p_idx}: \"{batch['prompts_text'][p_idx][:80]}...\"")
    print(f"{'─' * 60}")

    # 提取该组的数据
    start_idx = p_idx * G
    end_idx   = (p_idx + 1) * G

    group_rewards    = rewards[start_idx:end_idx]
    group_advantages = advantages[start_idx:end_idx]

    # 验证组内 statistics
    μ_reward = group_rewards.mean().item()
    σ_reward = group_rewards.std().item()
    μ_adv    = group_advantages.mean().item()  # 应该 ≈0
    σ_adv    = group_advantages.std().item()   # 应该 ≈1

    print(f"  Group stats:   μ_reward={μ_reward:.3f},  σ_reward={σ_reward:.3f}")
    print(f"                 μ_adv={μ_adv:.2e},   σ_adv={σ_adv:.3f}  ← μ_adv should be ≈0")

    for g_idx in range(G):
        sample_idx = start_idx + g_idx
        r = group_rewards[g_idx].item()
        a = group_advantages[g_idx].item()
        # 截断 response 显示
        resp_preview = responses[sample_idx][:100].replace("\n", " ")
        direction = "↑ BETTER" if a > 0 else ("↓ WORSE" if a < 0 else "→ avg")
        print(f"  [{g_idx}] reward={r:.2f}  adv={a:+.3f}  {direction}")
        print(f"       \"{resp_preview}...\"")

print(f"\n{'=' * 60}")
print("[Done] Phase 1 + 2 pipeline test complete.")
print("=" * 60)