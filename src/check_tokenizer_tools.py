"""
src/check_tokenizer_tools.py — 诊断 Qwen2.5 tokenizer 的 tool calling 能力
=========================================================================
在远程服务器上运行此脚本，查看 Qwen2.5-1.5B-Instruct 的 tokenizer 对工具调用的
原生支持情况，确定应该用原生 special tokens 还是需要自己添加。

Usage:
    python src/check_tokenizer_tools.py
"""

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# ---------------------------------------------------------------------------
# 工具定义（用于测试 apply_chat_template 的 tools 参数）
# ---------------------------------------------------------------------------
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算数学表达式的结果",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '3 + 5 * 2'",
                },
            },
            "required": ["expression"],
        },
    },
}


def main():
    print("=" * 70)
    print("Qwen2.5 Tokenizer Tool-Calling 诊断")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ===================================================================
    # 1. Chat Template 检查
    # ===================================================================
    print("\n[1] Chat Template 前 800 字符:")
    print("-" * 40)
    if tokenizer.chat_template:
        print(tokenizer.chat_template[:800])
    else:
        print("  ❌ 没有 chat_template！")

    # ===================================================================
    # 2. 检查是否已有 tool_call 相关的 special tokens
    # ===================================================================
    print("\n\n[2] Tool-related special tokens:")
    print("-" * 40)
    all_special = tokenizer.all_special_tokens
    tool_related = [t for t in all_special if "tool" in t.lower()]
    if tool_related:
        for t in tool_related:
            tid = tokenizer.convert_tokens_to_ids(t)
            print(f"  ✅ '{t}'  →  token_id = {tid}")
    else:
        print("  ⚠️  没有 tool 相关的 special tokens（需要自己添加）")

    # ===================================================================
    # 3. Tokenize tool_call 标签，看是否被切分
    # ===================================================================
    print("\n\n[3] Tokenization 粒度分析:")
    print("-" * 40)

    test_strings = [
        "<tool_call>",
        "</tool_call>",
        "<|tool_call_start|>",
        "<|tool_call_end|>",
        "<|tool_result|>",
        "<calc>",
        "</calc>",
        '<tool_call>{"name": "calculator", "arguments": {"expression": "3+5"}}</tool_call>',
    ]

    for s in test_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        tokens = [tokenizer.decode([tid]) for tid in ids]
        print(f"\n  '{s}'")
        print(f"    token_ids: {ids}")
        print(f"    tokens:    {tokens}")
        if len(ids) == 1:
            print(f"    ✅ 单个 token — 可以用 token ID 匹配")
        else:
            print(f"    ⚠️  {len(ids)} 个 token — 字符串匹配可能不对齐")

    # ===================================================================
    # 4. 测试 apply_chat_template 对 tools 参数的处理
    # ===================================================================
    print("\n\n[4] apply_chat_template 测试:")
    print("-" * 40)

    # --- 4a: 不带 tools 参数（现有方式） ---
    messages_without_tools = [
        {"role": "system", "content": "You have a calculator. Use <tool_call> to invoke it."},
        {"role": "user", "content": "What is 3 + 5?"},
    ]
    try:
        text_no_tools = tokenizer.apply_chat_template(
            messages_without_tools, tokenize=False, add_generation_prompt=True
        )
        print("\n  [4a] 不带 tools 参数:")
        print(f"  {text_no_tools!r}")
    except Exception as e:
        print(f"  [4a] ❌ 失败: {e}")

    # --- 4b: 带 tools 参数 ---
    messages_with_tools = [
        {"role": "user", "content": "What is 3 + 5?"},
    ]
    try:
        text_with_tools = tokenizer.apply_chat_template(
            messages_with_tools,
            tools=[CALCULATOR_TOOL],
            tokenize=False,
            add_generation_prompt=True,
        )
        print("\n  [4b] 带 tools 参数:")
        print(f"  {text_with_tools!r}")
    except Exception as e:
        print(f"  [4b] ❌ 失败: {e}")

    # --- 4c: 完整的 tool-use 对话（带 tool_call 和 tool 角色） ---
    full_tool_messages = [
        {"role": "system", "content": "You have a calculator tool."},
        {"role": "user", "content": "What is 3 + 5?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "3+5"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "8"},
        {"role": "assistant", "content": "The answer is 8."},
    ]
    try:
        text_full = tokenizer.apply_chat_template(
            full_tool_messages, tokenize=False, add_generation_prompt=False
        )
        print("\n  [4c] 完整 tool-use 对话（assistant tool_calls + tool 角色）:")
        print(f"  {text_full!r}")
    except Exception as e:
        print(f"  [4c] ❌ 失败: {e}")

    # ===================================================================
    # 5. 检查 tokenizer 是否支持 tool 角色
    # ===================================================================
    print("\n\n[5] 角色支持检查:")
    print("-" * 40)
    try:
        # 看 chat_template 源码里有没有 tool 或 tool_call 的字样
        has_tool_role = "tool" in tokenizer.chat_template.lower()
        has_tool_call = "tool_call" in tokenizer.chat_template.lower()
        print(f"  chat_template 包含 'tool' 角色逻辑: {has_tool_role}")
        print(f"  chat_template 包含 'tool_call' 逻辑: {has_tool_call}")
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")

    # ===================================================================
    # 6. 建议
    # ===================================================================
    print("\n\n[6] 建议:")
    print("-" * 40)
    print("  根据以上输出判断：")
    print("  - 如果 [2] 中有 tool_call special tokens → 直接用原生方案")
    print("  - 如果 [4b] 或 [4c] 成功 → apply_chat_template 原生支持 tool calling")
    print("  - 如果 [3] 中标签被切成多个 token → 需要 tokenizer.add_special_tokens()")
    print("  - 什么原生支持都没有 → 回退到自定义标签方案（当前方式）")


if __name__ == "__main__":
    main()
