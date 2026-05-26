import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Qwen2.5 的 tokenizer 默认没有 pad_token，我们手动设一个

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (this may take a few minutes the first time)...")

# 载入模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Model and tokenizer loaded successfully!")

prompt = "What is the capital of France?"

messages = [{"role": "user", "content": prompt}]

# 编码输入的prompt
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

# 模型生成输出
output = model.generate(inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
)

# 解码输出
response = tokenizer.decode(output[0], skip_special_tokens=True)

print("Response:", response)
print("Done!")