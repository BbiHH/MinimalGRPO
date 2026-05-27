import json

# 读取 tool_calling_prompts.jsonl 的前 100 条
with open('data/tool_calling_prompts.jsonl', 'r') as fin:
    lines = fin.readlines()[:100]

# 只保留 prompt 字段，写入 prompts.jsonl
with open('data/prompts.jsonl', 'w') as fout:
    for line in lines:
        data = json.loads(line)
        fout.write(json.dumps({'prompt': data['prompt']}) + '\n')

print('Done')