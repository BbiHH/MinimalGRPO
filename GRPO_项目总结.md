# MinimalGRPO 项目总结 —— LLM 后训练知识地图

> 这是一份从零构建 GRPO（Group Relative Policy Optimization）训练框架的完整工程记录。
> 面向读者：数月后重新打开这个仓库的自己，或任何想理解 LLM 后训练工程全貌的开发者。
> 文档定位：**知识词典 + 项目地图** —— 每个概念都指向具体代码位置，每条决策都记录了背后的 "为什么"。

---

## 目录

- [第一章：GRPO 算法理论基础](#第一章grpo-算法理论基础)
- [第二章：工程架构 —— 四阶段流水线](#第二章工程架构--四阶段流水线)
- [第三章：关键工程决策及其理由](#第三章关键工程决策及其理由)
- [第四章：工具使用训练 —— 从模拟到真实](#第四章工具使用训练--从模拟到真实)
- [第五章：监控与调试 —— 如何读懂训练曲线](#第五章监控与调试--如何读懂训练曲线)
- [第六章：单卡硬件工程](#第六章单卡硬件工程)
- [第七章：LLM 后训练知识地图](#第七章llm-后训练知识地图)
- [第八章：项目文件索引 —— 快速导航](#第八章项目文件索引--快速导航)
- [附录 A：完整训练配置参考](#附录-a完整训练配置参考)
- [附录 B：常见问题排查指南](#附录-b常见问题排查指南)
- [附录 C：后续扩展方向](#附录-c后续扩展方向)

---

## 第一章：GRPO 算法理论基础

### 1.1 LLM 对齐问题的数学表述

LLM 后训练（对齐）的核心问题可以表述为：在给定 prompt 分布下，最大化期望奖励，同时约束策略不要偏离参考模型太远。

$$
J(\theta) = \mathbb{E}_{x \sim D,\ y \sim \pi_\theta} \left[ R(x,y) - \beta \cdot D_{KL}\big(\pi_\theta(y|x) \parallel \pi_{ref}(y|x)\big) \right]
$$

**为什么需要 KL 约束？** 因为 reward 函数总是不完美的。如果不加约束，模型会学会输出在 reward 函数视角下得分高但人类不认可的文本——这叫做 *reward hacking*。KL 惩罚确保模型不会忘记预训练和 SFT 阶段学到的语言能力。

> **代码位置**：`src/loss.py:125-164` —— loss 函数中 `beta` 参数就是上面公式里的 $\beta$。

### 1.2 GRPO 的核心创新：组相对优势

**PPO 的依赖**：传统 PPO 需要一个 Value Function $V(s)$ 来计算 advantage $A = R - V(s)$。$V(s)$ 是一个独立的 Critic 网络，用来估计 "在当前状态下，后续能获得多少期望奖励"。

**Value Function 的三个问题**：
1. **参数开销**：需要额外的 Critic 网络，参数量翻倍（对 1.5B 模型来说就是额外 3GB 显存）
2. **训练不稳定**：Critic 的估计误差会传导到 Actor 的策略更新中
3. **状态定义模糊**：在语言模型中，"状态"是部分生成的文本序列，如何在 token 级别定义状态价值本身就很难

**GRPO 的解法**：对同一个 prompt，生成 $G$ 个不同的 response，然后在组内比较它们的 reward：

$$A_i = \frac{R_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}$$

直观理解：如果在同一个 prompt 下，response A 的得分比同组其他 response 都高，那 A 的生成策略就应该被强化；反之则抑制。组内均值 $\mu$ 天然充当了 baseline 的角色——它不需要额外的网络来学习。

> **代码位置**：`src/reward.py:273-338` —— `compute_advantages()` 的完整实现。

### 1.3 GRPO vs PPO vs DPO —— 三者边界

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| 是否需要 Critic 网络 | 是 | 否 | 否 |
| 是否需要在线生成 | 是 | 否（离线偏好对） | 是 |
| 优势信号来源 | $R - V(s)$ | 隐式（对数比） | $(R - \mu)/\sigma$ |
| 训练稳定性 | 中（Critic 可能不收敛） | 高（纯监督学习） | 中（依赖组内多样性） |
| 样本效率 | 高 | 低（一次学习） | 中（G 个 response 互相比较） |
| 硬件需求 | 高（Actor + Critic + Ref） | 低（单模型） | 中（Actor + Ref） |

**选择指南**：
- 有高质量偏好对数据集、不需要在线探索 → **DPO**
- 多卡集群、需要最大样本效率 → **PPO**（配合 Value Function）
- 单/双卡、希望训练 loop 内自闭环 → **GRPO**

> 本项目选择 GRPO 的直接原因：RTX 4090 单卡 24GB 无法承受额外 Critic 网络的显存开销。

### 1.4 GRPO 损失函数的逐项解剖

$$\mathcal{L} = -\frac{1}{|G|} \sum_i \left[ \min\left(\rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i\right) - \beta \cdot D_{KL}^i \right]$$

#### 项 1：重要性采样比率 $\rho_i = \exp(\log\pi_\theta - \log\pi_{ref})$

物理意义：当前策略相对 Reference 策略在某个 token 上的概率变化倍数。
- $\rho > 1$：模型对这个 token 变得更有信心（概率升高）
- $\rho < 1$：模型对这个 token 变得更保守（概率降低）
- $\rho \approx 0$：模型几乎放弃了 Reference 的选择

> **代码位置**：`src/loss.py:101-120` —— `compute_ratio_and_kl()`。

#### 项 2：PPO-clip $\min(\rho A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon) A)$

核心思想：限制单步策略更新的幅度，防止一次更新太激进导致训练崩溃。
- 当 $A > 0$（好事）：clip 上限 $1+\epsilon$，防止无限放大好行为的概率
- 当 $A < 0$（坏事）：clip 下限 $1-\epsilon$，防止无限压低坏行为的概率

> **代码位置**：`src/loss.py:139-145` —— 实现这两条 clip 分支。

#### 项 3：KL 惩罚 $\beta \cdot D_{KL}$

使用 **k3 近似公式**（二阶泰勒展开）：

$$D_{KL} \approx \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1$$

**为什么用近似而不是精确 KL？** 精确 KL 需要对整个词表（$V \approx 151936$）求和，计算量巨大。这个近似形式只需要一次 log-probs 求差，计算量减少了 $V$ 倍。

**关于 $\beta$ 的选择**：
- 太小 → reward hacking（模型学会骗 reward 函数）
- 太大 → 模型拒绝学习（策略被锁死在 Reference 附近）
- 本项目经验值：$\beta = 0.03 \sim 0.04$

> **代码位置**：`src/loss.py:116-118` —— KL 近似公式实现。

#### 项 4：序列长度归一化（最关键但最容易被忽视的细节）

这是本项目从 bug 中总结出的经验：

```python
# ❌ 错误做法：长文本主导梯度
loss = masked_loss.sum() / mask.sum()

# ✅ 正确做法：每个样本等权重
sample_losses = masked_loss.sum(dim=-1) / valid_lengths  # [N]
loss = sample_losses.mean()                               # scalar
```

如果不做长度归一化，长的 response 自然有更大的 loss 绝对值，会主导梯度方向。模型会学到 "越短越好" 来降低 loss，而非真正提升回复质量。

> **代码位置**：`src/loss.py:153-162` —— 长度归一的 loss 实现。

### 1.5 本章教训清单

- **G 不是越大越好**：$G=1$ 退化（无组比较），$G=128$ 显存爆炸。$G=4 \sim 8$ 是单卡合理区间
- **KL 近似在某些 token 上可能为负**（浮点误差），必须 `clamp(min=0)` —— 见 `src/loss.py:118`
- **$\beta$ 的选择依赖 reward 量级**：reward 绝对值越大，$\beta$ 需要相应调大
- **长度归一化不是可选的优化，是正确性要求**

---

## 第二章：工程架构 —— 四阶段流水线

### 2.1 宏观架构图

```
prompts[0..B-1]
  │
  ├─ Phase 1: Rollout (rollout.py)
  │   ├── load_training_data() → List[PromptExample]
  │   ├── tokenize_prompts()   → [B, L_prompt]
  │   ├── expand_for_group(G)  → [B*G, L_prompt]
  │   ├── generate_responses() → [B*G, L_total]
  │   └── extract_response()   → response_ids, response_mask
  │
  ├─ Phase 2: Reward (reward.py)
  │   ├── compute_rewards()    → [B*G]
  │   └── compute_advantages() → [B*G]
  │
  ├─ Phase 2.5: Metrics (main.py)
  │   └── tool/use_rate, tool/answer_accuracy, etc.
  │
  ├─ Phase 3: Reference Pre-compute (loss.py)
  │   └── compute_ref_log_probs() → ref_logp (cached)
  │
  └─ Phase 4: PPO Update Loop × 4 (loss.py + main.py)
      ├── compute_loss()        → loss scalar
      ├── loss.backward()
      ├── clip_grad_norm_(1.0)
      ├── optimizer.step()
      ├── KL check → early stop if > target_kl
      └── swanlab.log(metrics)
```

### 2.2 数据流与张量形状 —— 每一步的形状变化

这是整个项目最核心的数据流图。理解每一步的 shape 变化等于理解整个 pipeline：

```
Step 1: 数据加载
  load_training_data(cfg) → List[PromptExample], len = B
  文件: main.py:62-104

Step 2: Tokenize + Left-Padding
  tokenize_prompts(examples) → input_ids: [B, L_prompt]
  L_prompt = max(prompt_len_in_batch)，例如 50
  文件: src/rollout.py:110-181

Step 3: 扩展 B → B*G
  expand_for_group([B, L_prompt], G=8) → [B*G, L_prompt] = [16, 50]
  使用 torch.repeat_interleave(..., repeats=G, dim=0)
  文件: src/rollout.py:184-225

Step 4: 自回归生成
  model.generate([16, 50]) → full_ids: [16, L_total]
  L_total = L_prompt + max(response_len)，例如 50 + 180 = 230
  文件: src/rollout.py:228-294

Step 5: 提取 response
  extract_response_and_mask([16, 230], prompt_len=50) →
    response_ids:  [16, 180]
    response_mask: [16, 180]
  文件: src/rollout.py:297-334

Step 6: Reward 计算
  compute_rewards(texts) → rewards: [16]
  compute_advantages([16], G=8):
    内部 [16] → reshape → [2, 8] → 组内标准化 → reshape back → [16]
  文件: src/reward.py:36-81, src/reward.py:273-338

Step 7: Reference Log-Probs 预计算（一次性）
  compute_ref_log_probs(ref_model, batch) → logp_ref: [16, 180]
  文件: src/loss.py:80-95

Step 8: PPO 内循环（重复 4 次）
  每次：
    compute_log_probs(actor, batch) → logp_actor: [16, 180]
    compute_ratio_and_kl(actor, ref) → ratio: [16, 180], kl: [16, 180]
    compute_grpo_loss(ratio, advantages, kl) → loss: scalar
    loss.backward() → grad
    clip_grad_norm_(max_norm=1.0)
    optimizer.step()
```

> **代码位置**：`main.py:209-406` —— 完整的训练主循环，每步都对应上述流程。

### 2.3 模块职责划分

| 模块 | 文件 | 单一职责 | 核心函数 |
|------|------|---------|---------|
| 数据抽象 | `rollout.py:34-48` | PromptExample dataclass | `PromptExample` |
| 数据加载 | `rollout.py:62-107` | JSONL → PromptExample[] | `load_prompts()` |
| Tokenize | `rollout.py:110-181` | 文本→token IDs + Left-Pad | `tokenize_prompts()` |
| 扩展 | `rollout.py:184-225` | B → B*G 复制 | `expand_for_group()` |
| 生成 | `rollout.py:228-294` | model.generate() | `generate_responses()` |
| 提取 | `rollout.py:297-334` | 分离 prompt/response | `extract_response_and_mask()` |
| Rollout 入口 | `rollout.py:337-466` | 串联以上 5 步 | `rollout()` |
| 奖励 | `reward.py:36-81` | 标量打分 | `compute_rewards()` |
| 优势 | `reward.py:273-338` | 组内标准化 | `compute_advantages()` |
| Reward 入口 | `reward.py:345-384` | 串联 reward+advantage | `compute_rewards_and_advantages()` |
| Log-Probs | `loss.py:24-73` | 前向+提取 log-prob | `compute_log_probs()` |
| Ref 预计算 | `loss.py:80-95` | ref 前向一次性 | `compute_ref_log_probs()` |
| Ratio+KL | `loss.py:101-120` | 比率与 KL 近似 | `compute_ratio_and_kl()` |
| GRPO Loss | `loss.py:126-164` | PPO-clip + KL 损失 | `compute_grpo_loss()` |
| Loss 入口 | `loss.py:170-226` | 串联以上 4 步 | `compute_loss()` |
| 工具定义 | `tool.py:1-122` | 安全计算器 | `calculator()`, `execute_calcs()` |

### 2.4 Batch Dict —— 整个 pipeline 的数据总线

`batch` dict 是贯穿四个阶段的唯一数据载体。理解它的每个 key 等于理解整个数据流：

```python
batch = {
    # ===== Phase 1: Rollout 产出 =====
    "full_ids":        [B*G, L_total],       # 完整序列 (prompt + response + padding)
    "full_mask":       [B*G, L_total],       # 完整 attention mask
    "prompt_ids":      [B*G, L_prompt],      # 扩展后的 prompt
    "response_ids":    [B*G, L_response],    # 仅 response 部分
    "response_mask":   [B*G, L_response],    # response 有效 token mask (1=real, 0=pad)
    "prompt_len":      int,                  # prompt 统一长度
    "prompts_text":    List[str], len=B,     # 原始 prompt 文本（未扩展）
    "responses_text":  List[str], len=B*G,   # 生成的 response 文本
    "answers":         List, len=B*G,        # 标准答案（含 None）
    "task_types":      List[str], len=B*G,   # "calculator" | "general"
    "G":               int,                  # 组大小

    # ===== Phase 2: Reward 产出 =====
    "rewards":         [B*G],                # 每个样本的标量奖励
    "advantages":      [B*G],                # 组内标准化后的优势值

    # ===== Phase 3+4: Loss 产出 =====
    "logp_ref":        [B*G, L_response],    # Reference 模型的 log-probs（预计算，静态）
    "logp_actor":      [B*G, L_response],    # Actor 模型的 log-probs（每次 PPO step 更新）
    "ratio":           [B*G, L_response],    # 重要性采样比率
    "kl_per_token":    [B*G, L_response],    # per-token KL 散度近似值
    "loss":            scalar,               # 最终 GRPO 损失
    "entropy_mean":    float,                # 策略熵均值
    "ratio_mean":      float,                # 比率均值
    "ratio_max":       float,                # 比率最大值
}
```

### 2.5 本章教训清单

- **Dict 作为数据总线是最简单的正确方案**：不需要 protobuf、不需要 Ray Dataset。每个阶段往 dict 里写新 key，下游读取。唯一的风险是 key 名拼写错误——靠 assert/in 检查防御
- **文本列表和张量保持不同维度**：`prompts_text` 长度是 B，`responses_text` 长度是 B*G——文本只在 reward 和日志阶段用，不参与张量计算
- **不要在 batch dict 里放大对象到 GPU**：`responses_text` 是字符串列表，只放在 CPU。仅在 reward 计算时解码一次

---

## 第三章：关键工程决策及其理由

### 3.1 Left-Padding —— 自回归生成正确性的前提

**问题**：HuggingFace tokenizer 默认 `padding_side="right"`，但自回归生成必须 left-padding。

**原因**：
- Causal LM 从序列的第一个非 pad token 开始逐 token 生成
- Right-padding 时，不同长度 prompt 的结束位置不对齐 → 生成的起始位置也不对 → 短的 prompt 会在生成位置前有 padding token，破坏因果注意力
- Left-padding 确保所有样本的 prompt 都紧贴着生成起始位置

**实现要点**：
- `src/rollout.py:144` —— `tokenizer.padding_side = "left"` 必须在 tokenize 之前设置
- `src/rollout.py:173` —— tokenize 内部再次确认左侧填充，防止 HuggingFace 某些操作自动恢复默认值
- 仅仅在 `tokenize_prompts()` 里设置不够，因为调用链中某些操作可能重置 padding_side

### 3.2 Token-ID StoppingCriteria —— O(1) 终止条件

**问题**：工具调用场景下，模型输出 `</tool_call>` 时必须立即停止生成，但字符串匹配有 subword 边界问题。

**两种方案对比**：

| 方案 | 复杂度 | 可靠性 | 工业界采用 |
|------|--------|--------|-----------|
| 字符串匹配 | $O(L)$ per token decode + 搜索 | subword 可能切分标签导致匹配失败 | 否 |
| Token-ID 匹配 | $O(1)$ 检查 `input_ids[-1] == 151658` | 100% 可靠 | 是（vLLM/SGLang 内部都用） |

**验证步骤**（通过 `src/check_tokenizer_tools.py:86-93` 确认）：
```
'<tool_call>'  → token_ids: [151657]  ← 单个 token！
'</tool_call>' → token_ids: [151658]  ← 单个 token！
```

**实现**：`src/tool_use_poc.py:113-128` —— `StopOnToolCallEnd` 类，只需 6 行核心代码。

### 3.3 Length-Bias Fix —— GRPO Loss 的公平性修正

**问题**：朴素实现中长 response 有更大的 loss 绝对值 → 主导梯度方向 → 偏差。

**解决方案**：每个样本先 `mean` 再跨 batch `mean`。

```python
# ❌ 有偏（长文本权重高）
loss = masked_loss.sum() / mask.sum()

# ✅ 无偏（每个样本等权重）
sample_losses = masked_loss.sum(dim=-1) / valid_lengths  # [N]
loss = sample_losses.mean()                               # scalar
```

> **代码位置**：`src/loss.py:153-162`。

### 3.4 Reference Log-Probs 预计算 —— PPO 内循环的关键优化

**问题**：PPO 内循环每个 epoch 都需要计算 actor + ref 的 log-probs，但 ref 是冻结的，它的 log-probs 在 4 个 PPO epoch 期间不会改变。重复前向传播是浪费。

**优化方案**：在 PPO 内循环之前调用一次 `compute_ref_log_probs(ref_model, batch)`，存入 `batch["logp_ref"]`，之后循环内只对 actor 做前向传播。

**收益估算**（1.5B 模型，PPO epoch=4）：
- 节省 3 次 ref 前向传播
- 每次 ref 前向约 30ms → 每步节省 90ms
- 每 epoch (100 steps) 节省 9 秒

> **代码位置**：`src/loss.py:80-95`（预计算函数），`main.py:325-326`（调用点）。

### 3.5 Per-Example System Prompt —— 混合训练的基础设施

**旧架构的问题**：system_prompt 是 batch 级参数，所有样本共用一个。无法在同一 batch 中混合 calculator 任务（需要工具说明）和 general 任务（不需要工具说明）。

**解决方案**：引入 `PromptExample` dataclass，`system_prompt` 下沉为 per-example 字段：

```python
@dataclass
class PromptExample:
    text: str
    answer: Any = None
    task_type: str = "general"
    system_prompt: str | None = None   # ← 关键字段
```

**效果**：
- calculator 示例注入 `TOOL_SYSTEM_PROMPT`（教模型用 `<calc>` 标签）
- general 示例 `system_prompt = None`（保持通用对话能力）
- 同一 batch 内两种类型可以共存

> **代码位置**：`src/rollout.py:34-48`（dataclass 定义），`main.py:80-82`（注入 TOOL_SYSTEM_PROMPT），`src/rollout.py:150-168`（per-example 构建消息）。

### 3.6 行式存储 vs 列式存储

**旧设计（列式）**：
```python
prompts     = ["...", "..."]
answers     = [42, None]
task_types  = ["calc", "gen"]
# 添加新字段 → 修改所有函数签名
```

**新设计（行式）**：`PromptExample` 对象承载一行数据的所有字段。添加新字段只需修改 dataclass 定义。

**权衡**：行式存储更灵活，但在批量张量操作时需要解构成并行列表（`rollout.py:398-405`）。这里用一个 "内部解构" 层隔离了不匹配。

### 3.7 本章教训清单

- **不要过早做 KV-cache 优化**：本项目全程没有使用 `past_key_values`，每个 turn 重新 tokenize 完整历史。在探索阶段，正确性 > 性能
- **数据抽象的价值在 "数据加载和透传" 阶段**：PromptExample 在 tokenize 之前价值最大，进入张量计算后并行列表更自然。两者各司其职
- **Token-ID 匹配永远优于字符串匹配**：在 subword tokenization 的世界里，token ID 是唯一可靠的标识符

---

## 第四章：工具使用训练 —— 从模拟到真实

### 4.1 为什么在 GRPO 项目中做工具使用

GRPO 需要 reward 信号。工具使用提供的是**客观的、可自动化的 reward**：
- 通用对话的 reward 依赖启发式规则（`src/reward.py:140-190` `_heuristic_reward`），噪声大
- 数学计算题的 reward 依赖答案比对（`src/reward.py:93-138` `_tool_reward`），信号纯净

工具使用训练本质上是 "带约束的 GRPO"：模型不仅要输出正确答案，还要学会 "先调用工具再给出答案" 的推理范式。

### 4.2 方案一：自定义 `<calc>` 标签（模拟工具调用，当前训练用）

**协议设计**：
- 模型输出 `<calc>expression</calc>` 标记计算请求
- 模型输出 `<box>answer</box>` 标记最终答案
- 训练时**不实际执行计算器**——模型在一次生成中同时输出计算式和结果

**Reward 设计**（`src/reward.py:93-138`）：
```
格式分:  <calc> 正确闭合 +0.3,  <box> 正确闭合 +0.2  → 满分 0.5
结果分:  answer 正确 +1.0
工具分:  answer 正确 且 用了 calc → +0.3
满分: 1.8
```

**优点**：实现简单，不需要多轮生成，一次 `model.generate()` 出完整回复。

**缺点**：模型学的是 "在输出里假装调用工具"，对简单算术足够，但复杂推理时不可靠。

**System Prompt**：`src/rollout.py:53-60` —— `TOOL_SYSTEM_PROMPT`，告诉模型使用 `<calc>` 和 `<box>` 标签。

**数据**：`data/generate_tool_calling_data.py` —— 生成 2000 条加减乘除混合题，包含整数和混合运算。

### 4.3 方案二：Qwen2.5 原生 `<tool_call>` 格式（POC 验证完成，待接入训练）

**协议设计**：
- 使用 `tokenizer.apply_chat_template(messages, tools=[CALCULATOR_TOOL])` 自动生成 `<tools>` XML 定义和 tool-calling 指令
- 模型输出 JSON 格式的 `<tool_call>{"name": "calculator", "arguments": {...}}</tool_call>`
- 系统解析 JSON → 调用计算器 → 注入 `<tool_response>result</tool_response>` 作为 user 消息
- 多轮循环直到模型输出最终答案

**核心实现**：`src/tool_use_poc.py:220-414` —— `run_tool_use_loop()`，完整的独立多轮生成流程。

**关键组件**：
| 组件 | 位置 | 说明 |
|------|------|------|
| StopOnToolCallEnd | `tool_use_poc.py:113-128` | Token-ID 终止条件（ID 151658） |
| CALCULATOR_TOOL | `tool_use_poc.py:83-106` | OpenAI function calling schema 格式的工具定义 |
| parse_tool_calls() | `tool_use_poc.py:193-213` | 从生成文本中提取所有 tool_call JSON |

**多轮循环逻辑**（简化版）：
```python
for turn in range(MAX_TURNS):
    # Step 1: 构建带 tools 定义的 prompt
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tools=[CALCULATOR_TOOL],
        add_generation_prompt=True
    )
    # Step 2: 生成，遇到 </tool_call> 立即停止
    outputs = model.generate(
        stopping_criteria=[StopOnToolCallEnd(TOOL_CALL_END_ID)]
    )
    # Step 3: 检测 tool_call
    if has_tool_call(generated_text):
        parse JSON → call calculator() → inject <tool_response>
        continue  # 进入下一轮
    else:
        break  # 获得最终答案
```

### 4.4 两套方案的对比与选择

| 维度 | 自定义 `<calc>` 标签 | Qwen 原生 `<tool_call>` |
|------|---------------------|------------------------|
| 实现复杂度 | 低（一行 model.generate） | 高（多轮循环 + JSON 解析 + stop criteria） |
| 训练复杂度 | 低（不需要 actor_mask） | 中（需要 actor_mask 屏蔽工具返回 token） |
| 模型泛化性 | 仅限本项目 | 可迁移到任何支持 function calling 的框架 |
| 推理安全性 | 模拟工具调用 | 真实工具调用 |
| 适用场景 | 简单算术、快速实验 | 生产环境、多工具编排 |

### 4.5 Phase 5 未完成设计：真实交互式工具使用训练

**核心挑战**：
1. **Multi-turn Rollout**：将一次性 `model.generate()` 替换为多轮循环
2. **actor_mask 构建**：标记哪些 token 是模型生成的（参与 loss），哪些是工具注入的（不参与 loss）
3. **Loss 适配**：工具注入 token 参与 attention（模型需要看到计算结果），但不产生梯度
4. **Batch 处理**：不同样本的 turn 数和总长度不同，需要 per-sample padding
5. **KV-cache 管理**：是否复用上一轮的 KV-cache（性能 vs 实现复杂度）

> 详细设计见 `memory/project_tool_use_plan.md`。

### 4.6 Calculator 工具的安全实现

`eval()` 可以执行任意 Python 代码，安全风险极高。本项目的防御策略：

**双重安全防护**：
1. **正则预检**（`tool.py:61`）：`allowed_pattern` 只允许数字、运算符、括号、白名单函数名
2. **受限 eval**（`tool.py:70`）：`eval(expr, _SAFE_DICT, {})` —— 只暴露数学函数白名单，`__builtins__` 设为 None

**函数白名单**（`tool.py:12-18`）：
```python
_SAFE_DICT = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "int": int, "float": float,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "log": math.log, "log10": math.log10, "pi": math.pi, "e": math.e,
    "__builtins__": None,
}
```

### 4.7 本章教训清单

- **模拟工具调用是很好的起点，但不是终点**：它验证了 reward 设计，但模型学的是 "格式体操"
- **tokenizer 诊断是必须做的**：`src/check_tokenizer_tools.py` 不是辅助脚本——它决定用字符串匹配还是 token ID 匹配
- **多工具场景下，reward 需区分 "工具使用正确性" 和 "任务完成正确性"**

---

## 第五章：监控与调试 —— 如何读懂训练曲线

### 5.1 核心监控指标体系

#### 5.1.1 奖励信号类

| 指标 | SwanLab Key | 正常范围 | 异常信号 | 排查方向 |
|------|-----------|---------|---------|---------|
| 平均奖励 | `reward/mean` | 0.3~1.2 | < 0.1 长期不涨 | reward 设计太严或模型退化 |
| 奖励标准差 | `reward/std` | 0.2~0.6 | ≈ 0 | 所有 response 相同 → temperature 太低 |
| 组内奖励标准差 | `generation/within_group_reward_std` | > 0.1 | < 0.05 | 组内缺乏多样性，GRPO 信号弱 |

> **代码位置**：`main.py:232-248` —— reward 和 advantage 指标收集。

#### 5.1.2 策略更新类

| 指标 | SwanLab Key | 正常范围 | 异常信号 |
|------|-----------|---------|---------|
| 总 Loss | `loss/total` | -0.5~0.5 | 剧烈震荡（> 2.0 跳变）→ 学习率过大 |
| KL 散度 | `kl/mean` | 0.01~0.10 | > 0.15 → 策略偏离过快，可能即将崩溃 |
| Clip 比例 | `policy/clip_fraction` | 0.05~0.25 | > 0.4 → 太多 token 被 clip，更新过于激进 |
| 比率均值 | `policy/ratio_mean` | 0.8~1.2 | > 2.0 → 策略更新幅度过大 |
| 比率最大值 | `policy/ratio_max` | < 5.0 | > 10 → 极端重要性采样权重，梯度可能爆炸 |
| 策略熵 | `policy/entropy_mean` | 0.5~3.0 | < 0.2 → 策略过于确定，可能模式坍塌 |

> **代码位置**：`main.py:331-381` —— PPO 内循环中收集这些指标；`src/loss.py:197-209` —— ratio 和 entropy 的计算。

#### 5.1.3 生成长度与多样性类

| 指标 | SwanLab Key | 正常范围 | 异常信号 |
|------|-----------|---------|---------|
| 平均长度 | `generation/response_length_mean` | 30~150 | < 20 → 模型学会了 "越短越安全" |
| 最大长度 | `generation/response_length_max` | — | 经常达到 max_new_tokens → 需要增大上限 |
| 组内多样性 | `generation/response_distinct_2` | 0.5~1.0 | < 0.3 → 回复趋于同质化 |

> **代码位置**：`main.py:237-240`（长度），`main.py:307-320`（多样性）。

#### 5.1.4 工具使用类（calculator 任务专属）

| 指标 | SwanLab Key | 期望趋势 |
|------|-----------|---------|
| 工具使用率 | `tool/use_rate` | 训练过程中从 0 上升到 0.7+ |
| 答案准确率 | `tool/answer_accuracy` | 逐步提升至 0.8+ |
| 格式分均值 | `tool/format_score_mean` | 0.4 以上 |

> **代码位置**：`main.py:252-302` —— 工具使用相关的所有监控指标。

#### 5.1.5 训练系统类

| 指标 | SwanLab Key | 注意 |
|------|-----------|------|
| 学习率 | `train/lr` | 应遵循余弦衰减曲线 |
| 梯度范数 | `train/grad_norm` | 持续等于 max_norm → 可能被反复裁剪 |

> **代码位置**：`main.py:374-375`。

### 5.2 KL 散度早期停止机制

KL 散度是策略崩溃的**预警信号**。一旦 KL 超过阈值（`target_kl = 0.04`），说明 Actor 已偏离 Reference 太远，应立即停止当前 PPO 内循环。

```python
if valid_kl > config["target_kl"]:
    print(f"  [Early Stop] ppo_step {ppo_step+1}, KL={valid_kl:.4f}")
    break
```

> **代码位置**：`main.py:378-380`。

### 5.3 SwanLab 集成

SwanLab 是国产替代 WandB 的实验管理工具，API 与 WandB 几乎一致。在中国大陆网络环境下更稳定。

- 初始化：`main.py:176` —— `swanlab.init(project="MinimalGRPO", experiment_name="raw_2", config=config, mode="cloud")`
- 日志记录：`main.py:387` —— `swanlab.log(metrics)`，每个 step 记录一次
- 终端日志：`main.py:158-165` —— `log_metrics()`，简洁版输出，方便 AutoDL 命令行直接观察

### 5.4 Checkpoint 保存策略

两种格式，各有用处：

| 格式 | 路径 | 内容 | 用途 |
|------|------|------|------|
| 完整 checkpoint | `checkpoints/checkpoint_step_N.pt` | model + optimizer + scheduler state_dict | 断点续训 |
| HF 格式 | `checkpoints/model_step_N/` | config.json + model.safetensors | 直接加载推理 |

> **代码位置**：`main.py:143-155` —— `save_checkpoint()`。

### 5.5 本章教训清单

- **不要只看 loss**：loss 下降不代表模型变好。必须同时看 reward（是否上升）、KL（是否受控）、多样性（是否坍塌）
- **组内多样性是 GRPO 的命门**：`within_group_reward_std < 0.05` → 同一 prompt 的所有 response 几乎一样 → advantage 退化为噪声 → 应增大 temperature
- **clip_fraction 过高意味着学习率或 eps_clip 需要调整**：正常应该 < 0.25

---

## 第六章：单卡硬件工程

### 6.1 单卡显存预算分析

以 **Qwen2.5-1.5B-Instruct + GRPO** (B=2, G=8) 在 RTX 4090 24GB 上的显存分布：

| 组件 | 估算显存 | 说明 |
|------|---------|------|
| Actor 模型 (1.5B, bf16) | ~3.0 GB | 训练时需保持加载 |
| Reference 模型 (1.5B, bf16) | ~3.0 GB | 冻结但需加载 |
| AdamW 优化器状态 | ~6.0 GB | 一阶矩 + 二阶矩 = 参数量 × 2 (fp32) |
| 前向激活 (B*G=16, L=250) | ~2.0 GB | attention 中间结果 |
| batch 张量 (full_ids 等) | ~0.1 GB | 相对较小 |
| **合计** | **~14.1 GB** | 24 GB 卡上有充足余量 |

实际观测：显存峰值约 16-18 GB（模型加载时），稳定训练时约 12-14 GB。

### 6.2 bf16 vs fp16

| 格式 | 指数位 | 尾数位 | 动态范围 | LLM 训练 |
|------|--------|--------|---------|---------|
| fp16 | 5 位 | 10 位 | ±65,504 | 需要 loss scaling，容易溢出 |
| bf16 | 8 位 | 7 位 | 与 fp32 相同 | 不需要 loss scaling，推荐 |

**结论**：LLM 训练优先选 bf16，因为 loss scaling 的坑在 bf16 下不存在。

> **代码位置**：`main.py:57` —— `"dtype": torch.bfloat16`。

### 6.3 CUDA Cache 管理

长时间训练中，PyTorch 的 CUDA 内存分配器会产生碎片，导致即使空闲显存充足也会 OOM。

```python
def clear_cuda_cache(step):
    if step % 5 == 0:              # 每 5 步清理一次
        torch.cuda.empty_cache()   # 平衡性能与安全
```

> **代码位置**：`main.py:168-170`（定义），`main.py:406`（调用）。

**为什么不能每步都清？** `empty_cache()` 会释放所有缓存的 CUDA 内存，下一次分配需要重新申请 → 性能开销。5 步一次是经验平衡点。

### 6.4 梯度裁剪

GRPO 训练中，importance sampling ratio 可能导致某些 token 的梯度极大 → 单步更新破坏已学知识。

```python
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=1.0
)
```

> **代码位置**：`main.py:370`。

**监控**：如果 `grad_norm` 持续等于 `max_norm`，说明梯度始终被裁剪，可能需要调大 `max_norm` 或降低学习率。

### 6.5 为什么没有采用这些优化

#### Gradient Checkpointing
- **是什么**：正向传播不保存中间激活，反向传播时重新计算 → 用 20% 计算换 50% 显存
- **为什么不**：1.5B 模型在 24 GB 卡上显存充足（峰值 18 GB），不需要；compute 是瓶颈而非 memory

#### vLLM
- **是什么**：高性能推理引擎，吞吐是 HuggingFace generate 的 5-10 倍
- **为什么不**：单卡上 vLLM 模型 (~3GB) + HF actor (~3GB) + HF ref (~3GB) + optimizer (~12GB) + KV cache ≈ 23.5 GB → 24 GB 卡几乎爆满，随时 OOM

#### Gradient Accumulation
- **是什么**：累积多个小 batch 的梯度来模拟大 batch 训练
- **为什么不**：本项目 B=2, G=8 已经够用，不需要模拟更大 batch

### 6.6 本章教训清单

- **优先用 bf16，不要碰 fp16 的 loss scaling 坑**
- **`empty_cache()` 是双刃剑**：清得太频伤性能，不清太久 OOM。5 步一次是合理折中
- **Gradient Checkpointing 不是免费的午餐**：它用 compute 换 memory。memory 不是瓶颈时不要开

---

## 第七章：LLM 后训练知识地图

### 7.1 LLM 训练三阶段

```
Pre-training (预训练)
  —— 在海量文本上做 next-token prediction
  —— 产出：Base Model（会续写但不会对话）
    │
    ▼
Supervised Fine-Tuning (监督微调，SFT)
  —— 在 (prompt, ideal_response) 对上做交叉熵损失
  —— 产出：Instruct Model（会对话但存在安全/偏好问题）
    │
    ▼
Alignment / Post-Training (对齐/后训练)  ← 本项目的位置
  —— 让模型的输出更符合人类偏好
  —— 方法：RLHF, DPO, GRPO, Constitutional AI, ...
  —— 产出：Aligned Model（安全、有帮助、诚实）
```

### 7.2 GRPO 在 RLHF 家族中的位置

```
RLHF (Reinforcement Learning from Human Feedback)
  │
  ├── PPO-based (经典路线)
  │     └── 需要：Reward Model + Value Function (Critic) + Actor
  │     └── 代表：InstructGPT, ChatGPT 早期版本
  │     └── 与本项目关系：GRPO 是 PPO 的简化变体
  │
  ├── DPO (Direct Preference Optimization)
  │     └── 需要：偏好对数据集 (chosen, rejected)
  │     └── 不需要：在线生成、Reward Model、Value Function
  │     └── 代表：Zephyr, Llama3 的部分对齐
  │     └── 与本项目关系：DPO 是离线方法，GRPO 是在线方法
  │
  ├── GRPO (Group Relative Policy Optimization)  ← 本项目
  │     └── 需要：在线生成 G 个 response
  │     └── 不需要：Value Function（组内比较代替）
  │     └── 代表：DeepSeek-R1, DeepSeekMath
  │     └── 核心优势：去掉了 Critic 网络的额外开销
  │
  └── 其他变体：
        ├── KTO (Kahneman-Tversky Optimization)
        ├── IPO (Identity Preference Optimization)
        ├── SimPO (Simple Preference Optimization)
        └── ORPO (Odds Ratio Preference Optimization)
```

### 7.3 DeepSeek-R1 与本项目的关系

DeepSeek-R1 是 GRPO 算法最知名的成功案例。它使用 GRPO 训练 DeepSeek-V3-Base 获得了强推理能力。

Key insight：GRPO 的 group-relative advantage 非常适合推理任务——同一问题让模型生成多种推理路径，好的路径自然在组内得分更高。

本项目的 MinimalGRPO 与 R1 的差距：
| 维度 | MinimalGRPO | DeepSeek-R1 |
|------|------------|-------------|
| 模型规模 | 1.5B | 671B (MoE) |
| Reward 类型 | 规则 reward | 规则 + 模型 reward |
| 推理范式 | 单轮工具调用 | 长链思维推理 |
| 训练算力 | 单卡 RTX 4090 | 数千卡集群 |

但算法核心是一致的：**组采样 + 组内比较 + PPO-clip + KL 约束**。

### 7.4 Constitutional AI (CAI) 与 GRPO 的互补

Constitutional AI（Anthropic, 2023）的核心思想：让模型自己 "反思" 输出是否符合宪法原则，然后根据原则修改。

与 GRPO 的互补关系：
- CAI 解决 **"如何定义好的输出"**（通过宪法原则生成 reward）
- GRPO 解决 **"如何训练模型朝好的方向优化"**（通过组相对比较）
- 理想组合：CAI 原则生成 reward → 替代 `_heuristic_reward` → GRPO 做策略优化

### 7.5 概念到代码的快速索引

| 概念 | 代码位置 | 一句话说明 |
|------|---------|-----------|
| Left-Padding | `rollout.py:143-173` | 自回归生成的前置条件，padding 必须在左边 |
| Repeat Interleave | `rollout.py:221-222` | B → B*G 的关键操作 |
| Group Advantage | `reward.py:273-338` | GRPO 核心创新：组内标准化代替 Critic |
| Log-Softmax + Gather | `loss.py:56-62` | 提取实际 token 的 log-prob |
| KL 近似 (k3) | `loss.py:116-118` | 二阶泰勒展开，避免全词表求和 |
| PPO-Clip | `loss.py:139-145` | 限制单步策略更新幅度 |
| Length-Normalized Loss | `loss.py:153-162` | 消除序列长度偏置 |
| Per-Example System Prompt | `rollout.py:34-48` | 混合训练的基础设施 |
| Token-ID StoppingCriteria | `tool_use_poc.py:113-128` | O(1) 终止条件 |
| Safe Calculator (受限 eval) | `tool.py:12-81` | 白名单 + 正则双重防护 |
| Cosine Annealing | `main.py:137-139` | 学习率平滑衰减 |
| KL Early Stopping | `main.py:378-380` | 策略崩溃预警 |
| CUDA Cache Cleanup | `main.py:168-170` | 定期整理显存碎片 |

### 7.6 工业界后训练栈全景（本项目覆盖范围）

```
完整工业后训练栈                       MinimalGRPO 覆盖情况
─────────────────────────────────    ─────────────────────
Reward Model Training (RM)           ❌ 未覆盖（使用规则 reward）
Data Flywheel (数据飞轮)              ❌ 未覆盖
SFT Baseline                         ✅ 基座模型 Qwen2.5-Instruct
Online RL Training (PPO/GRPO)        ✅ 核心能力
Multi-Turn Tool Use                  ✅ POC 完成，训练设计中
Verifier / Reward Aggregation        ⚠️ 简单规则 reward，未做多 reward 融合
Distributed Rollout (vLLM/SGLang)    ❌ 单卡限制
Distributed Training (FSDP/DeepSpeed) ❌ 单卡限制
Production Deployment                ❌ 仅保存 HF 格式 checkpoint
```

### 7.7 本章教训清单

- **GRPO 不是银弹**：它在推理任务上出色（推理质量的组内区分度高），但在开放式对话任务上效果有限（对话质量难以用简单规则 reward 区分）
- **后训练是系统工程**：算法代码通常只占 20% 的工作量，数据构造、reward 设计、实验追踪、超参调优各占 20%

---

## 第八章：项目文件索引 —— 快速导航

### 8.1 核心文件

| 文件 | 行数 | 核心职责 | 最重要的函数/类 |
|------|------|---------|---------------|
| `main.py` | 419 | 训练主循环 + SwanLab 集成 | `main()`, `load_training_data()` |
| `src/rollout.py` | 466 | 数据加载 + 生成 | `PromptExample`, `rollout()`, `tokenize_prompts()` |
| `src/reward.py` | 384 | 奖励计算 + 组内优势 | `compute_rewards_and_advantages()`, `_tool_reward()` |
| `src/loss.py` | 226 | Log-probs + KL + PPO Loss | `compute_loss()`, `compute_grpo_loss()`, `compute_log_probs()` |
| `src/tool.py` | 122 | 安全计算器 | `calculator()`, `extract_calc_expressions()` |
| `src/tool_use_poc.py` | 477 | Qwen 原生工具调用 POC | `StopOnToolCallEnd`, `run_tool_use_loop()` |

### 8.2 辅助文件

| 文件 | 用途 |
|------|------|
| `src/check_tokenizer_tools.py` | 诊断 Qwen2.5 tokenizer 的 tool calling 原生支持能力 |
| `data/generate_tool_calling_data.py` | 生成 2000 条数学计算训练数据 |
| `data/tool_calling_prompts.jsonl` | 工具调用训练数据（2000 条） |
| `data/prompts.jsonl` | 通用对话训练数据（100 条，无答案） |
| `main_1_raw.py` | 旧版训练循环（StepLR 调度器，用于对比学习） |
| `test_phase1_2.py` | Rollout + Reward 阶段全链路验证 |
| `test_load.py` | 模型加载基础功能测试 |

### 8.3 关键设计决策索引

| 决策 | 代码/文档位置 |
|------|-------------|
| 为什么 left-padding | `src/rollout.py:117-118` 注释 |
| 为什么 per-example system_prompt | `src/rollout.py:25-31` dataclass 设计注释 |
| 为什么 length-normalized loss | `src/loss.py:153` 注释 |
| 为什么 ref log-probs 预计算 | `src/loss.py:85-86` docstring |
| 为什么混合训练 | `main.py:68-71` 注释 |
| Phase 5 真实工具调用训练设计 | `memory/project_tool_use_plan.md` |
| 为什么不用 verl/vLLM | `.claude/agent-memory/grpo-tech-director/migration-decision.md` |

---

## 附录 A：完整训练配置参考

### A.1 已验证可用的配置（Qwen2.5-1.5B-Instruct, RTX 4090 24GB）

```python
CONFIG = {
    # --- 模型 ---
    "model_name":         "Qwen/Qwen2.5-1.5B-Instruct",

    # --- Rollout ---
    "calculator_prompt_file":  "data/tool_calling_prompts.jsonl",
    "general_prompt_file":     "data/prompts.jsonl",
    "use_mixed_training":      None,          # True = 混合训练
    "batch_size":              2,
    "G":                       8,             # 每个 prompt 生成 8 个 response
    "max_new_tokens":          200,
    "temperature":             0.8,

    # --- Loss ---
    "eps_clip":           0.2,
    "beta":               0.03,              # KL 惩罚系数
    "ppo_epochs":         4,
    "target_kl":          0.04,              # KL 早期停止阈值

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
```

### A.2 配置项灵敏度分析

| 参数 | 作用 | 偏大后果 | 偏小后果 |
|------|------|---------|---------|
| `G` | 组内比较样本数 | 显存线性增长 | 组内统计不稳定，advantage 噪声大 |
| `temperature` | 采样多样性 | 回复质量下降（随机性过高） | 组内缺乏多样性，GRPO 信号弱 |
| `eps_clip` | PPO 更新限制幅度 | 更新过于保守，学不动 | 策略剧烈波动，可能崩溃 |
| `beta` | KL 约束强度 | 模型拒绝学习（被锁在 Reference） | reward hacking（学会骗奖励） |
| `target_kl` | KL 早停阈值 | 单次 PPO epoch 内可更新更多 | 策略偏离小就停，学得太慢 |
| `learning_rate` | 参数更新步长 | 训练不稳定，KL 飙升 | 训练速度太慢，reward 几乎不变 |

---

## 附录 B：常见问题排查指南

| 现象 | 可能原因 | 排查步骤 |
|------|---------|---------|
| loss 持续为 0 | advantage 全为 0 | 检查 `within_group_reward_std` 是否 < 0.01 |
| reward 不涨 | reward 设计不合理或模型退化 | 打印几个样本的 response + reward 分解，看哪个维度没给分 |
| KL 持续上升 | beta 太小 | 逐步增大 beta（0.04→0.06→0.08）直到 KL 稳定在 0.05 以下 |
| OOM | 显存碎片或 batch 太大 | 减小 B 或 G，或手动 `empty_cache()` |
| tokenizer 对 tool_call 标签切分错误 | 未做 tokenizer 诊断 | 运行 `python src/check_tokenizer_tools.py` 确认 token ID |
| 所有 response 相同 | temperature 太低或模式坍塌 | 增大 temperature 到 1.0，检查 `policy/entropy_mean` |
| loss 为 NaN | 梯度爆炸或数值不稳定 | 检查 `ratio_max` 是否极大，降低学习率，确保 `clamp(min=0)` 在 KL 计算中 |
| 模型学会输出极短回复 | 长度归一化未正确实现 | 检查 `loss.py:153-162` 是否正确使用 `valid_lengths` 做均值 |

---

## 附录 C：后续扩展方向

### C.1 在 MinimalGRPO 基础上的扩展

1. **Phase 5 — 真实交互式工具使用训练**
   - 将模拟工具调用升级为真正的多轮交互
   - 实现 actor_mask 屏蔽工具返回 token
   - 这是最有价值的下一步

2. **Replay Buffer（样本复用）**
   - 单次 rollout、多次训练的样本复用机制
   - 提升 GPU 利用率（减少生成时间占比）

3. **Multi-Tool 扩展**
   - 从 calculator 扩展到 search、code execution、web browsing
   - 验证多工具编排的 reward 设计和训练稳定性

4. **LLM-as-Judge Reward**
   - 用更强的模型（7B+）做 reward 评判
   - 覆盖开放式对话任务（当前只支持规则 reward）

5. **Dual-GPU + vLLM**
   - GPU0 做推理（vLLM），GPU1 做训练（PyTorch）
   - 推理和训练流水线化，吞吐大幅提升

6. **Process Reward Model (PRM)**
   - 给推理链的每一步打分，不只看最终结果
   - DeepSeek-R1 的核心技术之一

### C.2 扩展学习路径

7. **Verl 源码阅读**
   - 理解工业级 RL 训练框架的架构设计
   - 对比自建 pipeline 与工业框架的设计取舍
   - 不迁移代码，纯学习架构思想

8. **Constitutional AI 实验**
   - 用宪法原则替代启发式 reward
   - 结合 GRPO 做策略优化

---

> **写在最后**：这个项目从第一行 `tokenizer.padding_side = "left"` 开始，到完整的工具使用 GRPO 训练流程结束，覆盖了 LLM 后训练的完整闭环。它的价值不在于模型效果（1.5B 参数注定了上限），而在于**对每一个张量形状、每一次前向传播、每一个设计决策的深度理解**。这些理解将构成下一阶段工作的坚实基础。

> 项目时间线：2025年11月 - 2026年6月
> 硬件环境：AutoDL RTX 4090 24GB 单卡
> 基座模型：Qwen2.5-0.5B-Instruct → Qwen2.5-1.5B-Instruct
