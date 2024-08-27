import openai
from openai import OpenAI
import ollama
import threading
import yaml
from langchain.prompts import PromptTemplate
import time

        
class PromptGenerator:
    def __init__(self,
                 api_in_use: str,
                 openai_api_set: dict,
                 ollama_api_set: dict, 
                 style_template: list[dict],
                 sys_ins: str):
        
        self.sys_ins = sys_ins
        self.api_in_use = api_in_use
        self.threads = []
        self.Lock = threading.Lock()
        
        self.result_dict: dict[str, str] = {}
        self.full_process: dict[str, list] = {}
        
        if api_in_use == "openai":
            self.init_openai_api(openai_api_set)
        elif api_in_use =="ollama":
            self.init_ollama_api(ollama_api_set)
            
        self.load_style_template(style_template)
    
    def init_openai_api(self, api_set: dict):
        # self.api_key = api_set['api_key']
        # self.base_url = api_set['base_url']
        # self.model = api_set['model']
        self.__dict__.update(api_set)
        openai._client = OpenAI(
            api_key=self.api_key, # 替换成真实的API_KEY
            base_url=self.base_url, # 服务endpoint
        )
    
    def init_ollama_api(self, api_set: dict):
        self.__dict__.update(api_set)
        ollama._client = ollama.Client(
            host=f"http://{self.host}:{self.port}"
        )
        
    
    def load_style_template(self, style_template: list[dict]):
        self.prompt_template_dict = {}
        for style in style_template:
            self.prompt_template_dict[style['name']] = style['template']
        pass
    
    def single_response_openai(self, prompt: str)->str:
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.sys_ins},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    
    def single_response_ollama(self, prompt: str)->str:
        client = ollama.Client(
            host=f"http://{self.host}:{self.port}"
        )
    
        res = client.chat(
            model=self.model,
            stream=False,
            messages=[
                {"role": "system", "content": self.sys_ins},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": self.temperature}
        )
        return res['message']['content']
    
    def multi_turn_response_openai(self, raw_input: str, style_name: str, prompt_list: list[PromptTemplate])->str:
        input_var = {'input': raw_input}
        self.full_process[style_name] = []
        
        for i, temp in enumerate(prompt_list):
            prompt = temp.format(**input_var)
            res = self.single_response_openai(prompt)
            input_var = {'input': res}
            
            self.full_process[style_name].append(
                {f"turn {i+1}": {"prompt": prompt, "response": res}}
            )
        
        return res
    
    def multi_turn_response_ollama(self, raw_input: str, style_name: str, prompt_list: list[PromptTemplate])->str:
        input_var = {'input': raw_input}
        self.full_process[style_name] = []
        
        for i, temp in enumerate(prompt_list):
            prompt = temp.format(**input_var)
            res = self.single_response_ollama(prompt)
            input_var = {'input': res}
            
            self.full_process[style_name].append(
                {f"turn {i+1}": {"prompt": prompt, "response": res}}
            )
        
        return res
    
    def multi_thread_response(self, raw_input: str)->tuple[dict, dict]:
        self.result_dict = {}
        self.full_process = {}
        
        for style_name, temp_list in self.prompt_template_dict.items():
            if self.api_in_use == 'openai':
                tmp_thread = ResThreadOpenai(style_name, raw_input, temp_list, self)
            elif self.api_in_use == 'ollama':
                tmp_thread = ResThreadOllama(style_name, raw_input, temp_list, self)
            tmp_thread.start()
            self.threads.append(tmp_thread)
            
        for t in self.threads:
            t.join()
        
        return self.result_dict, self.full_process
        

class ResThreadOpenai(threading.Thread):
    def __init__(self, style: str, raw_input: str, prompt_list: list, 
                 generator: PromptGenerator):
        threading.Thread.__init__(self)
        self.style = style
        self.raw_input = raw_input
        self.prompt_list = prompt_list
        self.generator = generator
 
    def run(self):
        print(f"元素[{self.style}]开始", self.name)
        res = self.generator.multi_turn_response_openai(self.raw_input, self.style, self.prompt_list)
        self.generator.Lock.acquire()
        self.generator.result_dict[self.style] = res
        self.generator.Lock.release()
        
    def __del__(self):
        print(f"元素[{self.style}]结束", self.name)
        
class ResThreadOllama(threading.Thread):
    def __init__(self, style: str, raw_input: str, prompt_list: list, 
                 generator: PromptGenerator):
        threading.Thread.__init__(self)
        self.style = style
        self.raw_input = raw_input
        self.prompt_list = prompt_list
        self.generator = generator
 
    def run(self):
        print(f"元素[{self.style}]开始", self.name)
        res = self.generator.multi_turn_response_ollama(self.raw_input, self.style, self.prompt_list)
        self.generator.Lock.acquire()
        self.generator.result_dict[self.style] = res
        self.generator.Lock.release()
        
    def __del__(self):
        print(f"元素[{self.style}]结束", self.name)



if __name__=="__main__":
    t1 = time.time()
    with open("/home/roo/dream/wutr/mprompt-encoder/config/templates.yml", 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    styles = data['styles']
    api_in_use = data['api_in_use']
    ollama_api_set = data['ollama_api_set']
    openai_api_set = data['openai_api_set']
    sys_ins = data['sys_ins']
    gener = PromptGenerator(api_in_use, openai_api_set, ollama_api_set, styles, sys_ins)
    res, his = gener.multi_thread_response("一个红色的苹果")
    # print(res.keys())
    # print(his)
    print("最终结果\n")
    for k, v in res.items():
        print(f"元素: {k}")
        print(v)
        print('\n')
    
    print("中间过程\n")
    for k, v in his.items():
        print(f"元素: {k}")
        for i, s in enumerate(v):
            print(f"turn {i+1}:")
            print(f"prompt:\n{s[f'turn {i+1}']['prompt']}")
            print(f"response:\n{s[f'turn {i+1}']['response']}")
        print('\n')
        
    print(time.time()-t1)