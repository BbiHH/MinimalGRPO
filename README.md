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
* 额外：实现一次标准的模型训练实验，构造 数学计算任务，需要使用 tool call 来调用计算器实现 加减乘除运算。
* 构建 实验数据 、 工具调用 、 设计新的reward。 添加 tool.py 。 


### Phase 5: Response Reuse & Engineering Optimizations (`src/replay_buffer.py`)

GRPO 的核心样本效率来自一次采样、多次学习。Phase 4 的 naive 实现中，每次参数更新都会重新 rollout，这会把大部分 GPU 时间消耗在生成而非训练上。Phase 5 的目标是将采样与学习解耦。

#### 5.1 核心改动

* 实现 `ResponseReplayBuffer` 类，负责缓存 rollout 产出的 `(full_ids, advantages, response_mask)` 等张量。
* 单次 rollout 产出的 $B \times G$ 个 response 可以被训练循环复用 `num_reuse_epochs` 次。
* 每次复用前对 advantage 做小幅度噪声注入（可选），防止过拟合。

#### 5.2 额外工程优化

* **Gradient Accumulation**：通过累积小 batch 的梯度来模拟更大的训练 batch，降低显存压力。
* **Gradient Checkpointing**：利用 `torch.utils.checkpoint` 在反向传播时重建中间激活，用时间换显存。
* **Dynamic Padding**：按 batch 内最大长度动态 padding，而非固定长度，减少无效计算。

#### 5.3 Replay Buffer 接口设计

```python
# src/replay_buffer.py — 伪代码框架
from collections import deque
from typing import Dict
import torch

class ResponseReplayBuffer:
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, rollout_batch: Dict[str, torch.Tensor]):
        """缓存一次 rollout 的所有张量（转移到 CPU 以节省 GPU 显存）"""
        cpu_batch = {k: v.cpu() for k, v in rollout_batch.items()
                     if isinstance(v, torch.Tensor)}
        self.buffer.append(cpu_batch)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """随机采样一个 mini-batch 用于梯度更新"""
        ...

    def clear(self):
        """全量刷新（新 rollout 时调用）"""
        self.buffer.clear()
```

#### 5.4 执行验证

* 修改 `main.py` 训练循环：Rollout → `buffer.add()` → `for epoch in reuse_epochs: train_step()`
* 验证指标：对比有无 reuse 时的 Reward 提升速度和 GPU 利用率。

---

### Phase 6: Online Experiment Monitoring (`wandb` / `swanlab`)

真实训练需要实时可视化，否则就是黑盒。Phase 6 的目标是接入在线实验管理工具，实现以下指标的实时追踪和曲线对比。

#### 6.1 监控指标体系

| 指标类别 | 具体指标 | 含义 |
|---------|---------|------|
| 奖励信号 | `reward/mean`, `reward/std`, `reward/min`, `reward/max` | 组内奖励分布 |
| KL 散度 | `kl/mean_per_token`, `kl/max_per_token` | 策略偏离程度 |
| 策略损失 | `loss/policy`, `loss/kl_penalty`, `loss/total` | 损失分解 |
| 比率统计 | `ratio/mean`, `ratio/clip_fraction` | PPO clip 触发比例 |
| 生成长度 | `response/mean_len`, `response/std_len` | 生成的 token 数分布 |
| 梯度信息 | `grad/norm`, `lr` | 梯度范数、学习率 |
| 系统资源 | `gpu/memory_used`, `gpu/utilization` | 显存和利用率 |

#### 6.2 接入步骤（以 swanlab 为例，国产替代 wandb）

```python
# main.py 中初始化
import swanlab

swanlab.init(
    project="minimal-grpo",
    config={
        "model": "Qwen2.5-0.5B-Instruct",
        "G": 4,
        "max_new_tokens": 128,
        "eps_clip": 0.2,
        "beta": 0.04,
        "lr": 1e-6,
    }
)

# 训练循环中记录
swanlab.log({
    "reward/mean": rewards.mean().item(),
    "kl/mean": kl_mean,
    "loss/total": loss.item(),
    "response/mean_len": response_lengths.mean().item(),
}, step=global_step)
```

#### 6.3 执行验证

* 启动训练后在浏览器打开 `swanlab` 或 `wandb` 仪表盘。
* 确认所有指标曲线正常更新，KL 散度应在训练初期上升后趋于平稳。

---

### Phase 7: vLLM Inference-Engine Separation (Dual-GPU)

当单卡同时承载 rollout 和训练时，生成速度严重受限（自回归生成是串行的）。Phase 7 使用 vLLM 作为独立推理引擎，在 GPU0 上做批次推理，在 GPU1 上做梯度训练，形成经典的 **Rollout Worker + Trainer Worker** 流水线。

#### 7.1 架构升级

```text
┌─────────────────┐         ┌──────────────────────┐
│   GPU 0: vLLM   │ ──────> │   GPU 1: Trainer     │
│   (Rollout)     │  batch  │   (PyTorch Training) │
│                 │ <────── │                      │
│   参数同步      │ weights │   梯度更新 + 回传    │
└─────────────────┘         └──────────────────────┘
```

#### 7.2 实现要点

1. **vLLM Server 启动**：在 GPU0 上以 API 模式启动 vLLM，加载与 Trainer 相同的基座模型。

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000 \
    --max-model-len 1024
```

2. **Rollout 改写**：将 `src/rollout.py` 中的 `model.generate()` 替换为对 vLLM API 的异步批量请求，使用 `asyncio` + `aiohttp` 并发获取 response。

3. **权重同步**：每个训练 step 结束后，将 Trainer 更新后的模型权重通过文件或共享内存同步给 vLLM 引擎。

4. **流水线执行**：
   * Rollout Worker 持续生成新 batch 并推入 replay buffer
   * Trainer Worker 从 buffer 取数据做梯度更新
   * 两者异步并行，吞吐大幅提升

#### 7.3 执行验证

* 使用 `nvidia-smi` 确认两张 GPU 同时有负载（一张做推理，一张做训练）。
* 对比 Phase 6 单卡方案和 Phase 7 双卡方案的 **每秒有效训练步数**，应有 $3\sim5\times$ 提升。

---

## 6. Future Directions

### 6.1 Model-as-Judge Reward (Phase 8)

当前 Phase 2 的 reward 是基于规则（长度、格式、关键词）的，对于开放式指令跟随任务远远不够。Phase 8 引入 LLM-as-Judge 模式，使用一个更强的模型（如 GPT-4o-mini 或本地部署的 Qwen2.5-7B）作为评判器。

#### 实现思路

* **LLM Judge Prompt 模板**：构建包含评分标准和输出格式的系统 prompt。
* **批量评判**：每个训练 step 的 $B\times G$ 个 response 发送给 Judge 模型评分。
* **Reward 融合**：规则奖励（速度导向）和 Judge 奖励（质量导向）加权融合。

```text
SYSTEM: You are an impartial judge. Rate the following response on:
  1. Helpfulness (1-5)
  2. Accuracy (1-5)
  3. Coherence (1-5)
Output ONLY a JSON: {"helpfulness": X, "accuracy": Y, "coherence": Z}
```

#### 挑战与对策

| 挑战 | 对策 |
|------|------|
| Judge 调用延迟大 | 异步批量请求 + 缓存 |
| Judge 本身有 bias | 多个 Judge 模型交叉验证 |
| API 调用成本 | 优先部署本地 7B 模型，或在关键 milestone 才使用云端 API |

### 6.2 更丰富的 Reward 策略

* **Length Penalty**：防止模型学会无限延长回答来刷分。
* **Repetition Penalty**：对重复 n-gram 惩罚，提升多样性。
* **Task-Specific Reward**：数学题用 `\boxed{}` 正则提取答案比对；代码题用 `exec()` 执行验证。

---

## 7. Complete Project Roadmap

| Phase | 名称 | 核心产出 | 状态 |
|-------|------|---------|------|
| 1 | Rollout Verification | `src/rollout.py` + shape 对齐 | ✅ 已完成 |
| 2 | Reward & Advantage | `src/reward.py` + 组内标准化验证 | ✅ 已完成 |
| 3 | Core Backward | `src/loss.py` + log-probs / KL / PPO-clip | ✅ 已完成 |
| 4 | Training Loop | `main.py` + 显存管理 | ✅ 已完成 |
| 5 | Response Reuse & Optimization | `src/replay_buffer.py` + 多次复用 + 梯度累积 | 📋 待开发 |
| 6 | Online Monitoring | `wandb` / `swanlab` 集成 + 完整指标面板 | 📋 待开发 |
| 7 | vLLM Separation (Dual-GPU) | vLLM rollout worker + trainer worker 流水线 | 📋 待开发 |
| 8 | Model-as-Judge | LLM Judge reward + 规则融合 | 🔮 规划中 |

---

## 8. Quick Reference: Running Each Phase

```bash
# Phase 1+2: Rollout + Reward/Advantage 测试
python test_phase1_2.py

# Phase 1-4: 完整训练循环（测试用）
python main.py

# Phase 6+: 带监控的正式训练
python main.py --use_swanlab

# Phase 7+: 双卡分离模式
bash scripts/launch_vllm_worker.sh  # GPU 0
python main.py --trainer_only       # GPU 1
```