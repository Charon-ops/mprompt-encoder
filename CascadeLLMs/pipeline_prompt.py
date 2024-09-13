import yaml
import time
import os
import json
import threading
from tqdm import tqdm
import torch
from prompt_generator import PromptGenerator

lock = threading.Lock()

def get_raw_prompts(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw_prompts = f.readlines()
        
    return raw_prompts

def dict2jsonl(res: dict, path):
    with open(path, "a") as f:
        line = json.dumps(res)
        f.write(line+'\n')
        

class  MultiThreadPrompt(threading.Thread):
    def __init__(self, idx: int, raw_prompts: list[str], **kwargs):
        threading.Thread.__init__(self)
        self.idx = idx
        self.raw_prompts = raw_prompts
        self.gener = PromptGenerator(**kwargs)
        
    def run(self):
        pro_bar = tqdm(self.raw_prompts, ncols=100,
                       desc=f"CUDA {self.idx}",
                       position=self.idx,
                       ascii=False
                       )
        for raw_p in pro_bar:
            res, his = self.gener.multi_thread_response(raw_p)
            # summ, summ_prompt = gener.get_summary(res) # summary
            res['raw_prompt'] = raw_p
            # res['summary'] = summ
            lock.acquire()
            dict2jsonl(res, output_path)
            lock.release()
        pro_bar.close()
        



if __name__=="__main__":
    # t1 = time.time()
    prompt_path = '../prompt'
    output_path = '../prompt/T2I_val.json'
    raw_prompts = []
    prompt_list = []
    files = os.listdir(prompt_path)
    
    # read all raw prompts
    for file in files:
        path = os.path.join(prompt_path, file)
        raw_prompts += get_raw_prompts(path)
    
    # split by cuda num
    num_cuda = torch.cuda.device_count()
    num_per_cuda = len(raw_prompts) // num_cuda
    res = len(raw_prompts) % num_cuda
    for i in range(num_cuda):
        prompt_list.append(raw_prompts[i*num_per_cuda: (i+1)*num_per_cuda])
    if res > 0:
        for i, j in enumerate(raw_prompts[-res:]):
            prompt_list[i] += j
    
    # load cfg
    with open("../config/templates_en_new.yml", 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    styles = data['styles']
    summary = data['summary']
    api_in_use = data['api_in_use']
    ollama_api_set = data['ollama_api_set']
    openai_api_set = data['openai_api_set']
    sys_ins = data['sys_ins']
    
    # multi thread
    threads = []
    port = int(ollama_api_set['port'])
    for i in range(num_cuda):
        print(f"Starting CUDA {i}")
        ollama_api_set['port'] = str(port+i)
        tmp_thread = MultiThreadPrompt(i,
                                       prompt_list[i],
                                       api_in_use, 
                                       openai_api_set, 
                                       ollama_api_set, 
                                       styles, 
                                       summary, 
                                       sys_ins
                                       )
        tmp_thread.start()
        threads.append(tmp_thread)
    for t in threads:
        t.join()
        
    
    # gener = PromptGenerator(api_in_use, openai_api_set, ollama_api_set, styles, summary, sys_ins)

    
        
    # print(time.time()-t1)