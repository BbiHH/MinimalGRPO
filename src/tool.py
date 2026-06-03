"""
src/tool.py — 可被模型调用的工具集合
==============================
目前只包含一个安全的计算器工具。
工具接口：输入字符串，输出字符串结果。
"""

import re
import math

# 允许的计算函数白名单（防止 eval 执行恶意代码）
_SAFE_DICT = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "int": int, "float": float,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "log": math.log, "log10": math.log10, "pi": math.pi, "e": math.e,
    "__builtins__": None,
}

# 计算器表达式解析
def extract_calc_expressions(text: str) -> list[str]:
    """
    从文本中提取所有 <calc>...</calc> 标签内的表达式。

    使用非贪婪匹配，只提取正确闭合的标签内容（必须有 </calc>）。
    不处理未闭合或嵌套的标签。

    参数：
        text: 可能包含 <calc> 标签的文本

    返回：
        表达式字符串列表（不包含标签本身）

    示例：
        extract_calc_expressions("<calc>3+5</calc> and <calc>2*4</calc>")
        -> ["3+5", "2*4"]
    """
    # 非贪婪匹配成对出现的 <calc>...</calc>
    pattern = r'<calc>(.*?)</calc>'
    return re.findall(pattern, text)


# 计算器工具
def calculator(expression: str) -> str:
    """
    安全计算数学表达式，返回字符串结果。
    
    支持的运算：+ - * / // % ** 以及白名单中的函数。
    如果表达式无效或执行出错，返回错误信息。
    
    示例：
        calculator("3 + 5 * 2")  -> "13"
        calculator("sqrt(16)")   -> "4.0"
    """
    # 移除空白
    expr = expression.strip()
    if not expr:
        return "[Error: empty expression]"

    # 基本安全检查：只允许数字、运算符、括号、白名单函数名
    allowed_pattern = r'^[\d\s\+\-\*\/\%\.\(\)\,\^\_a-zA-Z]+$'
    if not re.match(allowed_pattern, expr):
        return "[Error: invalid characters in expression]"

    # 将 ^ 替换为 ** (幂运算)
    expr = expr.replace("^", "**")

    try:
        # 使用受限的 namespace 执行 eval
        result = eval(expr, _SAFE_DICT, {})
        # 格式化结果，整数显示为整数，浮点数保留合理精度
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 6)
        return str(result)
    except SyntaxError:
        return "[Error: syntax error]"
    except Exception as e:
        return f"[Error: {str(e)}]"
    
# 计算器工具编排
def execute_calcs(text: str) -> tuple[str, list[str]]:
    """
    执行文本中所有 <calc>expression</calc> 的计算，
    返回替换后的文本和结果列表。

    对每个 <calc> 标签内的表达式调用 calculator() 计算，
    然后将 "<calc>expr</calc>" 替换为 "<calc>expr</calc> = result"。

    参数：
        text: 包含 <calc> 标签的文本

    返回：
        (modified_text, results) 元组
        - modified_text: 每个标签被替换为带结果的文本
        - results: 各表达式计算结果（字符串）

    示例：
        text = "Let me calculate. <calc>3+5</calc> The answer is 8."
        execute_calcs(text)
        -> ("Let me calculate. <calc>3+5</calc> = 8 The answer is
8.", ["8"])
    """
    pattern = r'<calc>(.*?)</calc>'

    # 取出所有表达式并计算
    expressions = re.findall(pattern, text)
    results = [calculator(expr) for expr in expressions]

    # 逐个替换，确保每个标签对应正确的计算结果
    counter = 0

    def replace_match(match: re.Match) -> str:
        nonlocal counter
        expr = match.group(1)
        result = results[counter]
        counter += 1
        return f"<calc>{expr}</calc> = {result}"

    modified_text = re.sub(pattern, replace_match, text)
    return modified_text, results