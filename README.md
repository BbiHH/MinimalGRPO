# Minimal GRPO Implementation for LLM Alignment

## 1. Project Overview
本项目是一个从零构建的极简组相对策略优化（Group Relative Policy Optimization, GRPO）强化学习框架。核心目标是在单/双卡 GPU 环境下，通过原生 PyTorch 张量操作实现大语言模型（LLM）的对齐训练，屏蔽高度封装的第三方 RL 库（如 TRL），以确立对底层数据流、张量形状（Tensor Shape）以及显存调度的系统性工程掌控。

## 2. Environment & Specifications
*   **Framework**: PyTorch, Hugging Face `transformers`
*   **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` (或同级小参数量模型)
*   **Precision**: `torch.bfloat16`
*   **Hardware Constraint**: AutoDL 单节点 (RTX 4090 )
*   **Padding Strategy**: 强制 `Left-Padding` 以支持自回归生成


## 3. Directory Structure
```text
minimal-grpo/
├── data/
│   └── prompts.jsonl         # 训练使用的基础 Prompt 集
├── src/
│   ├── rollout.py            # 数据并行采样与输入输出张量构建
│   ├── reward.py             # 基于规则/正则的奖励函数计算
│   ├── loss.py               # Log-Probs 提取、KL 计算与策略损失函数
│   └── utils.py              # 显存清理与监控埋点
├── main.py                   # 训练主循环入口
└── README.md

```

## 4. Algorithmic Implementation Details

### 4.1 Rollout & Generation

针对 Batch Size 为 $B$ 的输入，在张量维度扩充组大小 $G$：

1. 提取 Prompt 并 Tokenize 得到 `input_ids`。
2. 使用 `torch.repeat_interleave(input_ids, repeats=G, dim=0)`，使得批次扩展为 $B \times G$。
3. 调用 `model.generate(do_sample=True, ...)` 生成响应序列。

### 4.2 Advantage Standardization

基于同一个 Prompt 生成的 $G$ 个回复计算奖励得分 $R$，并在组内进行标准化计算优势值（Advantage）：

$$\hat{A}_{i}=\frac{R_i-\mu_R}{\sigma_R+\epsilon}$$

* $\mu_R$: 组内得分均值
* $\sigma_R$: 组内得分标准差
* $\epsilon$: $10^{-4}$，防止除零错误

### 4.3 Log-Probs & KL Divergence

计算当前 Actor 模型与冻结权重的 Reference 模型的对数概率。使用以下公式近似计算 Token 级别的 KL 散度：

$$D_{\text{KL}}\approx\frac{\pi_{\text{ref}}}{\pi_\theta}-\log\frac{\pi_{\text{ref}}}{\pi_\theta}-1$$

### 4.4 Policy Objective

基于 PPO 裁剪机制的 GRPO 变体损失函数：

$$\mathcal{L}=-\frac{1}{G}\sum_{i=1}^G\left[\min\left(\rho_i\hat{A}_i,\text{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat{A}_i\right)-\beta D_{\text{KL}}\right]$$

* 其中 $\rho_i=\exp(\log\pi_\theta-\log\pi_{\text{ref}})$ 为重要性采样比率。

---

## 5. Execution Pipeline (For AI Agent Execution)

如果使用 AI Agent 辅助开发，请严格遵循以下增量开发顺序：

### Phase 1: Rollout Verification (`src/rollout.py`)

* 实现加载模型与 Tokenizer 的逻辑。
* 验证 `input_ids` 的 `repeat_interleave` 操作与 `generate` 输出张量的 shape 对齐。

### Phase 2: Reward & Advantage (`src/reward.py`)

* 实现标量奖励函数并验证张量化后的 Advantage 组内零均值属性。

### Phase 3: Core Backward (`src/loss.py`)

* 实现 `gather` 操作提取生成 Token 的真实 Log-Probs。
* 实现 Actor 与 Reference 的 KL 惩罚与 Loss 计算。

### Phase 4: Training Loop (`main.py`)

* 集成上述模块，加入 `torch.cuda.empty_cache()` 控制显存。
* 集成 TensorBoard/WandB 记录 Mean Reward, KL Divergence, Policy Loss。