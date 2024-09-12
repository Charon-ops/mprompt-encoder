import json
import os

def dict2jsonl(res: dict, path):
    with open(path, "a") as f:
        line = json.dumps(res)
        f.write(line+'\n')

with open("0test80.txt", 'r') as f:
    lines = f.readlines()
    
with open("test80.jsonl", "r") as file:
    for i, line in enumerate(file):
        json_obj = json.loads(line)
        json_obj['raw_prompt'] = lines[i]
        # print(json_obj)
        dict2jsonl(json_obj, "test801.jsonl")
        # assert 0
