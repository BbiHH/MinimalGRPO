"""
生成用于 工具调用 实验的数学问题数据。
输出格式：每行一个 JSON 对象，包含 "prompt" 和 "answer"。
"""
import json
import random
import os

random.seed(42)

OUTPUT_FILE = "data/tool_calling_prompts.jsonl"
NUM_SAMPLES = 2000

def generate_sample():
    """生成一个数学问题 + 正确答案"""
    type_ = random.choice(["add", "sub", "mul", "div", "mix"])

    if type_ == "add":
        a = random.randint(10, 9999)
        b = random.randint(10, 9999)
        templates = [
            f"What is {a} plus {b}?",
            f"Compute the sum: {a} + {b}.",
            f"I have {a} apples and buy {b} more. How many apples do I have in total?",
            f"Sarah has ${a} in her savings account. She deposits another ${b}. What is her new balance?",
            f"A stadium has {a} seats in the lower bowl and {b} seats in the upper bowl. What is the total capacity?",
        ]
        answer = a + b

    elif type_ == "sub":
        a = random.randint(100, 9999)
        b = random.randint(10, a)  # ensure positive
        templates = [
            f"What is {a} minus {b}?",
            f"Subtract {b} from {a}.",
            f"A store had {a} items in stock. After selling {b}, how many are left?",
            f"John had ${a}. He spent ${b}. How much money does he have now?",
            f"The temperature was {a}°C and dropped by {b}°C. What is the new temperature?",
        ]
        answer = a - b

    elif type_ == "mul":
        a = random.randint(10, 999)
        b = random.randint(10, 99)
        templates = [
            f"Calculate {a} multiplied by {b}.",
            f"What is the product of {a} and {b}?",
            f"A box contains {a} packets. Each packet has {b} candies. How many candies in total?",
            f"A car travels at {a} km/h for {b} hours. How far does it go?",
            f"If a ticket costs ${a}, how much do {b} tickets cost?",
        ]
        answer = a * b

    elif type_ == "div":
        # integer division with remainder
        b = random.randint(2, 50)
        a = b * random.randint(10, 200)  # exact multiple
        templates = [
            f"Divide {a} by {b}.",
            f"What is {a} / {b}?",
            f"A farmer has {a} eggs to pack into cartons of {b}. How many cartons can he fill?",
            f"A rope of length {a} cm is cut into {b} equal pieces. How long is each piece?",
            f"${a} is to be shared equally among {b} people. How much does each get?",
        ]
        answer = a // b  # integer division

    else:  # mix: two operations
        a = random.randint(10, 500)
        b = random.randint(10, 100)
        c = random.randint(10, 100)
        op1 = random.choice(["+", "-"])
        op2 = random.choice(["*", "//"])
        # avoid division by zero
        if op2 == "//" and c == 0:
            c = 1

        templates = [
            f"Calculate: {a} {op1} {b} {op2} {c}.",
            f"First {'add' if op1=='+' else 'subtract'} {a} and {b}, then {'multiply' if op2=='*' else 'divide'} by {c}.",
            f"({a} {op1} {b}) {op2} {c} = ?",
        ]

        # compute with Python
        expr = f"({a} {op1} {b}) {op2} {c}"
        answer = eval(expr)  # safe because all numbers

        if op2 == "//":  # make sure it's integer
            answer = int(answer)

    prompt = random.choice(templates)
    return {"prompt": prompt, "answer": answer, "task_type": "calculator"}


# 生成数据
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for _ in range(NUM_SAMPLES):
        sample = generate_sample()
        f.write(json.dumps(sample) + "\n")

print(f"Generated {NUM_SAMPLES} samples to {OUTPUT_FILE}")