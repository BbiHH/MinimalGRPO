'''
Docstring for projects.MinimalGRPO.src.rollout
本模块负责：
  1. 从文件加载 Prompts
  2. 对 Prompts 进行 Left-Padding 和 Tokenize
  3. 将每个 Prompt 复制 G 份（组大小）用于 GRPO 组内比较
  4. 调用模型生成 Response
  5. 整理并返回所有需要的张量，供后续 reward.py 和 loss.py 使用

设计原则：
  - 纯函数式，不引入额外抽象层级
  - 每个关键步骤都有形状注释，方便追踪数据流
  - 强制 Left-Padding，确保自回归生成的正确性
'''

import json
import torch
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field


# ============================================================
# 数据抽象层：PromptExample — 训练数据的基本单元
# ============================================================
# 设计理念：
#   原来的代码用三个并行列表 (prompts, answers, task_types) 承载数据，
#   这种"列式存储"(columnar) 在添加新字段时需要修改所有函数签名和调用处。
#   改为"行式存储"(row-oriented) 后，每一行数据是一个自包含的 PromptExample，
#   添加新字段只需修改 dataclass 定义，调用方无需改动 —— 随数据流自然透传。
#
#   另一个关键设计：system_prompt 从 batch 级参数下沉为 per-example 字段。
#   这让混合训练成为可能：calculator 示例注入工具使用说明，general 示例不注入。
@dataclass
class PromptExample:
    """训练数据的基本单元，自包含所有字段，随数据流透传

    Attributes:
        text:           prompt 文本，必填字段
        answer:         标准答案（tool-use 任务）, None 表示无答案的任务
        task_type:      任务类型，决定 reward 分发逻辑 ("calculator" → 工具评分, "general" → 启发式评分)
        system_prompt:  per-example 的 system prompt，None 表示不注入 system message
                        这是混合训练的关键：calculator 示例注入 TOOL_SYSTEM_PROMPT，
                        general 示例不注入，让模型在同一个 batch 中学到不同的交互模式
    """
    text: str
    answer: Any = None
    task_type: str = "general"
    system_prompt: str | None = None


# 引入工具调用的prompt

TOOL_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a calculator. "
    "When you need to perform arithmetic, you MUST use the calculator "
    "by writing <calc>expression</calc>. Then provide the final answer. \n"
    "example: \n"
    "User: Compute the sum: 1545 + 3592."
    "Assistant: Let me use the calculator <calc>1545 + 3592</calc>, the answer is <box>5179</box>."
)

def load_prompts(filepath: str) -> List[PromptExample]:
    """
    功能: 从 JSONL 文件中加载 prompt 数据，返回 PromptExample 列表

    输入:
        filepath: str  —  e.g. "data/tool_calling_prompts.jsonl"

    输出:
        examples: List[PromptExample]  —  每条记录一个 PromptExample，
                  调用方可按需设置 system_prompt 等字段（如注入 TOOL_SYSTEM_PROMPT）

    JSONL 字段约定:
        "prompt"     (必填) — prompt 文本
        "answer"     (可选) — 标准答案，默认 None
        "task_type"  (可选) — 任务类型，默认 "general"

    实现:
        逐行读取 JSONL，每行构造一个 PromptExample。
        注意：system_prompt 不在此处设置 —— 由上层调用方（如 load_training_data）
        根据数据集类型注入，保持了"数据加载"和"训练配置"的职责分离。
    """
    examples: List[PromptExample] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:       # 跳过空行
                continue
            data = json.loads(line)
            example = PromptExample(
                text=data["prompt"],
                answer=data.get("answer", None),
                task_type=data.get("task_type", "general"),
                # system_prompt 保持 None，由上层调用方按需注入
            )
            examples.append(example)

    # 统计日志
    num_with_answers = sum(1 for ex in examples if ex.answer is not None)
    type_counts: Dict[str, int] = {}
    for ex in examples:
        type_counts[ex.task_type] = type_counts.get(ex.task_type, 0) + 1

    print(f"[rollout] Loaded {len(examples)} prompts from {filepath}")
    print(f"[rollout]   with answers (tool-use tasks): {num_with_answers}")
    print(f"[rollout]   task type distribution: {type_counts}")
    return examples


def tokenize_prompts(
    tokenizer,
    examples: List[PromptExample],
) -> Dict[str, torch.Tensor]:
    """
    功能: 将 PromptExample 列表转为 token ids，并做 Left-Padding

    为什么必须 Left-Padding?
        - 自回归模型从左到右逐 token 生成
        - 如果右侧有 padding token，模型生成时会产生混乱
        - Left-Padding 确保所有 padding 在序列左侧，生成从右侧开始

    关键设计变更（vs 旧版）:
        system_prompt 不再作为 batch 级参数传入，而是从每个 PromptExample
        的 .system_prompt 字段读取。这使得同一个 batch 内可以混合不同的
        system_prompt 配置 —— 例如 calculator 示例带工具说明，general 示例不带。

    输入:
        tokenizer: HuggingFace tokenizer 实例
        examples:  List[PromptExample], 长度 = B

    输出:
        prompt_dict: Dict 包含:
            "input_ids"      — shape [B, L_prompt]     (Left-Padded)
            "attention_mask" — shape [B, L_prompt]     (1=有效token, 0=padding)
        其中 L_prompt = max(prompt_len_in_this_batch)

    实现细节:
        1. 设置 tokenizer.padding_side = "left"
        2. 按 example.system_prompt 构建 chat 格式消息
        3. 用 apply_chat_template 生成格式化文本
        4. tokenizer(..., padding=True, return_tensors="pt") 自动 padding
    """
    # --- 步骤 1: 配置 left-padding ---
    tokenizer.padding_side = "left"      # 强制左侧填充

    # --- 步骤 2: 每个 example 按自身 system_prompt 包装成 chat 格式 ---
    # 关键设计：system_prompt 是 per-example 的，支持混合训练
    # - calculator 示例: system_prompt=TOOL_SYSTEM_PROMPT → 生成 [system, user] 消息
    # - general 示例:    system_prompt=None              → 生成 [user] 消息
    messages_list = []
    for ex in examples:
        if ex.system_prompt is not None:
            messages = [
                {"role": "system", "content": ex.system_prompt},
                {"role": "user", "content": ex.text},
            ]
        else:
            messages = [
                {"role": "user", "content": ex.text},
            ]
        messages_list.append(messages)

    # apply_chat_template 会生成类似:
    #   "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"
    # 这样的格式化字符串
    formatted_texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]

    # --- 步骤 3: Tokenize + Padding ---
    # padding=True 会自动把不同长度的序列 padding 到 batch 内最长序列的长度
    tokenizer.padding_side = "left"  # 再次确保
    prompt_dict = tokenizer(
        formatted_texts,
        padding=True,                 # 自动 Left-Padding（因为 padding_side="left"）
        truncation=False,             # 不截断，prompt 一般不太长
        return_tensors="pt",          # 返回 PyTorch 张量
    )

    B = len(examples)
    L_prompt = prompt_dict["input_ids"].shape[1]
    print(f"[rollout] Tokenized: B={B}, L_prompt={L_prompt}")
    print(f"[rollout] input_ids shape: {prompt_dict['input_ids'].shape}")

    return prompt_dict


def expand_for_group(
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    G: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    功能: 将每个 Prompt 复制 G 份，用于 GRPO 的组内采样
    
    为什么需要复制?
        GRPO 的核心思想：同一个 prompt，让模型生 G 个不同的 response，
        然后组内比较优劣。因此 prompt 需要扩展 B → B*G。
    
    输入:
        prompt_ids:      shape [B, L_prompt]
        attention_mask:  shape [B, L_prompt]
        G: int, 组大小 (e.g., G=4 表示每个 prompt 生成 4 个回复)
    
    输出:
        expanded_ids:     shape [B*G, L_prompt]
        expanded_mask:    shape [B*G, L_prompt]
    
    实现:
        torch.repeat_interleave(tensor, repeats=G, dim=0)
        — 在 batch 维逐元素重复 G 次。
        
        例如 B=2, G=3:
          [A, B]  →  [A, A, A, B, B, B]
        
        数据布局变为:
        batch_idx=0 → prompts[0] 的 G 个副本
        batch_idx=1 → prompts[0] 的 G 个副本 (不是! 应该是下一个prompt)
        
        更直观地:
        原始:  [prompt_0, prompt_1]
        扩展后: [prompt_0, prompt_0, prompt_0, prompt_1, prompt_1, prompt_1]
        (重复 G=3)
    """
    expanded_ids = torch.repeat_interleave(prompt_ids, repeats=G, dim=0)
    expanded_mask = torch.repeat_interleave(attention_mask, repeats=G, dim=0)
    
    B_orig = prompt_ids.shape[0]
    print(f"[rollout] Expanded: {B_orig} prompts × G={G} = {expanded_ids.shape[0]} samples")
    print(f"[rollout] expanded_ids shape: {expanded_ids.shape}")
    
    return expanded_ids, expanded_mask


@torch.no_grad()  # 生成阶段不需要梯度，节省显存
def generate_responses(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    功能: 调用模型自回归生成 Response
    
    输入:
        model:           HuggingFace CausalLM 模型实例
        tokenizer:       tokenizer (用于获取 pad_token_id)
        prompt_ids:      shape [B*G, L_prompt]  — expanded prompt tokens
        attention_mask:  shape [B*G, L_prompt]  — 对应 mask
        max_new_tokens:  int, 生成的最大新 token 数
        temperature:     float, 采样温度（越高越随机）
        top_p:           float, nucleus sampling 的 top-p 值
    
    输出:
        full_ids: shape [B*G, L_prompt + L_response]
                 完整序列 = prompt + response
    
    实现细节:
        1. 获取 pad_token_id（但 generate 内部会处理 padding...）
           - model.generate 使用 attention_mask 知道哪些是 padding，
             不会对 padding token 计算 attention。
        
        2. 采样参数 do_sample=True + temperature > 0 确保多样性。
           - 如果 temperature=0，会走 greedy decoding，失去多样性。
        
        3. 返回的序列包含 prompt + 新生成的 token。
    """
    model.eval()  # 确保是 eval 模式
    
    # 确保张量在正确设备上
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # --- 调用 generate ---
    # 参数含义：
    #   max_new_tokens:  最多新生成 token 数
    #   do_sample=True:  启用采样（而非 greedy）
    #   temperature:     控制分布锐度（<1 变尖锐，>1 变平坦）
    #   top_p=0.9:       nucleus sampling，保留累积概率前 90% 的 token
    #   pad_token_id:    用于处理 batch 中不同长度的 padding
    #   eos_token_id:    遇到 EOS 就停止该样本的生成
    full_ids = model.generate(
        prompt_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # full_ids shape: [B*G, L_prompt + actual_response_len]
    # 注意：不同样本的 actual_response_len 可能不同！
    # model.generate 会自动 padding 到最长序列
    print(f"[rollout] Generated: full_ids shape = {full_ids.shape}")
    
    return full_ids


def extract_response_and_mask(
    full_ids: torch.Tensor,
    prompt_len: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    功能: 从完整序列中分离出 prompt 和 response，并构建正确的 attention_mask
    
    为什么需要这个函数?
        model.generate 返回的是 padded 后的完整序列 [B*G, total_len]。
        我们需要：
          - response_ids: 只包含模型生成的部分
          - response_mask: 标记 response 中哪些是真实 token（非 padding/EOS）
          - 用于后续 reward 计算和 log-probs 提取
    
    输入:
        full_ids:       shape [B*G, L_total] — prompt + response + 可能的 padding
        prompt_len:     int — prompt 部分统一的长度（因为 left-padding 后相同）
        pad_token_id:   int — padding token 的 id
    
    输出:
        response_ids:   shape [B*G, L_response_max]
                        从 full_ids 中截取的 response 部分
        response_mask:  shape [B*G, L_response_max]
                        1=真实生成的token, 0=padding (用于后续 loss 屏蔽)
        full_attention_mask: shape [B*G, L_total]
                            完整序列的 attention_mask (prompt padding + response)
    """
    # --- 构建完整 attention_mask ---
    # full_ids 中 != pad_token_id 的位置为 1，其余为 0
    full_mask = (full_ids != pad_token_id).long()  # shape [B*G, L_total]
    
    # --- 截取 response 部分 ---
    # 因为 left-padding，所有样本的 prompt 长度相同 (= prompt_len)
    response_ids = full_ids[:, prompt_len:]        # shape [B*G, L_response]
    response_mask = full_mask[:, prompt_len:]       # shape [B*G, L_response]
    
    print(f"[rollout] Response shape: {response_ids.shape}")
    print(f"[rollout] Response mask shape: {response_mask.shape}")
    
    return response_ids, response_mask, full_mask


def rollout(
    model,
    tokenizer,
    examples: List[PromptExample] = None,
    prompt_file: str = None,
    G: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> Dict[str, torch.Tensor]:
    """
    功能: Rollout 主函数，串联上述所有步骤

    这是外部调用的唯一入口。一次调用完成：
      加载 → Tokenize → 扩展 → 生成 → 整理输出

    数据流说明:
      PromptExample 是贯穿整个 pipeline 的数据载体:
        prompt_file → load_prompts() → List[PromptExample]
                                              ↓
                              rollout() 内部解构成并行列表
                              (text/answer/task_type) 用于后续处理
                                              ↓
                              tokenize_prompts(examples) 按 per-example
                              system_prompt 构建 chat 消息
                                              ↓
                              返回 batch_dict（格式不变，兼容 reward.py/loss.py）

    参数:
        model:          HuggingFace CausalLM
        tokenizer:      HuggingFace tokenizer（已设 pad_token）
        examples:       List[PromptExample], 直接传入数据（推荐路径，来自 main.py）
        prompt_file:    str, e.g. "data/tool_calling_prompts.jsonl"（便捷路径，
                        内部自动调用 load_prompts）
        G:              int, 组大小 (每个 prompt 生成 G 个 response)
        max_new_tokens: int, 最大生成 token 数
        temperature:    float, 采样温度

    返回值:
        batch: Dict 包含以下 key:
            "full_ids"        — [B*G, L_total]      完整序列
            "full_mask"       — [B*G, L_total]      完整 attention_mask
            "prompt_ids"      — [B*G, L_prompt]     扩展后的 prompt
            "response_ids"    — [B*G, L_response]   仅 response 部分
            "response_mask"   — [B*G, L_response]   response 的有效 mask
            "prompt_len"      — int, prompt 统一长度
            "prompts_text"    — List[str], 原始 prompt 文本（B 个，非 B*G）
            "responses_text"  — List[str], 生成的 response 文本（B*G 个）
            "answers"         — List, 标准答案（B*G 个，含 None），用于 tool-use reward
            "task_types"      — List[str], 任务类型（B*G 个），用于 reward dispatch
            "G"               — int, 组大小
    """
    # Step 1: 获取 PromptExample 列表（两种路径）
    if prompt_file is not None:
        # 便捷路径：从文件加载（向后兼容，适合快速测试）
        examples = load_prompts(prompt_file)
    elif examples is not None:
        # 标准路径：调用方直接传入已配置好的 examples（如 load_training_data）
        pass
    else:
        raise ValueError("Either prompt_file or examples must be provided.")

    # Step 1.5: 从 PromptExample 列表中解构出并行列表（内部使用）
    # 为什么在内部解构成并行列表而不是修改下游函数签名？
    #   - reward.py / loss.py 的接口已稳定，不需要改动
    #   - 并行列表在 expand_for_group 和 compute_advantages 中更自然
    #   - PromptExample 的价值在"数据加载和透传"阶段，而非"张量计算"阶段
    prompts_text = [ex.text for ex in examples]
    answers = [ex.answer for ex in examples]
    task_types = [ex.task_type for ex in examples]

    B = len(examples)

    # Step 2: 扩展 answers 和 task_types B → B*G（与 prompt 扩展对齐）
    answers_expanded = []
    for a in answers:
        for _ in range(G):
            answers_expanded.append(a)

    task_types_expanded = []
    for t in task_types:
        for _ in range(G):
            task_types_expanded.append(t)

    # Step 3: Tokenize + Left-Padding（per-example system_prompt 在此生效）
    prompt_dict = tokenize_prompts(tokenizer, examples)
    prompt_ids = prompt_dict["input_ids"]        # [B, L_prompt]
    prompt_mask = prompt_dict["attention_mask"]   # [B, L_prompt]
    L_prompt = prompt_ids.shape[1]

    # Step 4: 扩展 B → B*G
    expanded_ids, expanded_mask = expand_for_group(prompt_ids, prompt_mask, G)
    # expanded_ids: [B*G, L_prompt]

    # Step 5: 生成 response
    full_ids = generate_responses(
        model, tokenizer,
        expanded_ids, expanded_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    # full_ids: [B*G, L_total]

    # Step 6: 提取 response 和 mask
    pad_token_id = tokenizer.pad_token_id
    response_ids, response_mask, full_mask = extract_response_and_mask(
        full_ids, L_prompt, pad_token_id
    )

    # Step 7: 解码文本（用于 reward 计算）
    responses_text = tokenizer.batch_decode(
        response_ids, skip_special_tokens=True
    )

    print(f"[rollout] Complete! Generated {len(responses_text)} responses.")
    print(f"[rollout] Sample response[0]: {responses_text[0][:100]}...")

    return {
        "full_ids":         full_ids,          # [B*G, L_total]
        "full_mask":        full_mask,         # [B*G, L_total]
        "prompt_ids":       expanded_ids,      # [B*G, L_prompt]
        "response_ids":     response_ids,      # [B*G, L_response]
        "response_mask":    response_mask,     # [B*G, L_response]
        "prompt_len":       L_prompt,
        "prompts_text":     prompts_text,       # List[str], len=B
        "responses_text":   responses_text,     # List[str], len=B*G
        "answers":          answers_expanded,      # List, len=B*G (含 None)
        "task_types":       task_types_expanded,    # List[str], len=B*G
        "G":                G,
    }