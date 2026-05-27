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