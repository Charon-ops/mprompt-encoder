import openai
from openai import OpenAI
import threading
import time

# 配置api-key
api_key="f2oFPKXs3EW0eJh7fhfBCTLRZtWOMym5i1vMfm7e5ENmk1DI" # 替换成真实的API_KEY
base_url="https://api.f2gpt.com/v1" # 服务endpoint
openai._client = OpenAI(
    api_key=api_key, # 替换成真实的API_KEY
    base_url=base_url, # 服务endpoint
)

# 参数
model = "gpt-4-turbo" # 模型
turns = 3 # 级联层数
num_style = 3 # 风格数，每个风格单独一个线程
style_list = ['1', '2', '3']
prompt_dict = {}
system_instruction_list = ["重复用户给你的内容，不要改变任何内容"]
result_dict = {}

tmp_prompt = "重复这个句子，不要改变或添加任何内容，编写一首诗，解释编程中的递归概念。"

# 全局变量
Lock = threading.Lock()
threads = []


def single_response(prompt: str)->str:
    client = OpenAI(
        api_key=api_key, # 替换成真实的API_KEY
        base_url=base_url, # 服务endpoint
    )
 
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_instruction_list[0]},
            {"role": "user", "content": prompt}
        ]
    )
    
    return completion.choices[0].message.content

def multi_turn_response(prompt: str)->str:
    res = prompt
    for i in range(turns):
        res = single_response(res)
        # print(f"{i} turn")
    
    return res

def multi_thread_response()->None:
    global result_dict
    result_dict = {}
    
    for i in range(num_style):
        tmp_thread = responseThread(style_list[i], tmp_prompt)
        tmp_thread.start()
        threads.append(tmp_thread)
        
    for t in threads:
        t.join()

class responseThread(threading.Thread):
    def __init__(self, style: str, prompt: str):
        threading.Thread.__init__(self)
        self.style = style
        self.prompt = prompt
 
    def run(self):
        print(f"风格[{self.style}]开始")
        res = multi_turn_response(tmp_prompt)
        Lock.acquire()
        result_dict[self.style] = res
        Lock.release()
        
    def __del__(self):
        print(f"风格[{self.style}]结束", self.name)



if __name__=="__main__":
    multi_thread_response()
    print(result_dict)
    pass