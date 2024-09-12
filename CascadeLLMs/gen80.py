import yaml
import time
import os
import json
import threading
from prompt_generator import PromptGenerator


def get_raw_prompts(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw_prompts = f.readlines()
        
    return raw_prompts

def dict2jsonl(res: dict, path):
    with open(path, "a") as f:
        line = json.dumps(res)
        f.write(line+'\n')

if __name__=="__main__":
    t1 = time.time()
    raw_prompts = get_raw_prompts("./0test80.txt")
    with open("/home/roo/dream/wutr/mprompt-encoder/config/templates_en.yml", 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    styles = data['styles']
    summary = data['summary']
    api_in_use = data['api_in_use']
    ollama_api_set = data['ollama_api_set']
    openai_api_set = data['openai_api_set']
    sys_ins = data['sys_ins']
    gener = PromptGenerator(api_in_use, openai_api_set, ollama_api_set, styles, summary, sys_ins)

    for raw_p in raw_prompts[:20]:
        res, his = gener.multi_thread_response(raw_p)
        summ, summ_prompt = gener.get_summary(res) # summary
        # print(summ_prompt+'\n\n')
        res['raw_prompt'] = raw_p
        res['summary'] = summ
        dict2jsonl(res, 'test2.jsonl')
        
    print(time.time()-t1)