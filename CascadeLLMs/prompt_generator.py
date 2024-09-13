import openai
from openai import OpenAI
import ollama
import threading
import yaml
from typing import Optional, Union
from langchain.prompts import PromptTemplate
import time

        
class PromptGenerator:
    def __init__(self,
                 api_in_use: str,
                 openai_api_set: dict,
                 ollama_api_set: dict, 
                 style_template: list[dict],
                 summary: str,
                 sys_ins: str):
        
        self.sys_ins = sys_ins
        self.api_in_use = api_in_use
        self.threads: list[threading.Thread] = []
        self.Lock = threading.Lock()
        
        self.result_dict: dict[str, str] = {}
        self.full_process: dict[str, list] = {}
        
        if api_in_use == "openai":
            self.init_openai_api(openai_api_set)
        elif api_in_use =="ollama":
            self.init_ollama_api(ollama_api_set)
            
        self.load_style_template(style_template)
        self.load_summary_template(summary)
    
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
            temp_list: list[PromptTemplate] = []
            for temp in style['template']:
                temp_list.append(
                    PromptTemplate(
                        input_variables=['input'],
                        template=temp
                    )
                )
            self.prompt_template_dict[style['name']] = temp_list
    
    def load_summary_template(self, summary: str):
        self.summary = PromptTemplate(
            input_variables=['artistic_style', 'composition', 'visual_center', 'color_tone'],
            template=summary
        )
        
    
    def single_response_openai(self, prompt: str, max_new_tokens: Optional[int]=None)->str:
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        if max_new_tokens is None:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.sys_ins},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": self.temperature}
            )
        else:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.sys_ins},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": self.temperature, "max_new_tokens": max_new_tokens}
            )
        return completion.choices[0].message.content
    
    def single_response_ollama(self, prompt: str, max_new_tokens: Optional[int]=None)->str:
        client = ollama.Client(
            host=f"http://{self.host}:{self.port}"
        )
        if max_new_tokens is None:
            res = client.chat(
                model=self.model,
                stream=False,
                messages=[
                    {"role": "system", "content": self.sys_ins},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": self.temperature}
            )
        else:
            res = client.chat(
                model=self.model,
                stream=False,
                messages=[
                    {"role": "system", "content": self.sys_ins},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": self.temperature, "max_new_tokens": max_new_tokens}
            )
        return res['message']['content']
    
    def multi_turn_response_openai(self, raw_input: str, style_name: str, prompt_list: list[PromptTemplate])->str:
        input_var = {'input': raw_input}
        self.full_process[style_name] = []
        length = len(prompt_list)
        for i, temp in enumerate(prompt_list):
            prompt = temp.format(**input_var)
            if i < length-1:
                res = self.single_response_openai(prompt)
            else:
                res = self.single_response_openai(prompt, 77)
            input_var = {'input': res}
            
            self.full_process[style_name].append(
                {f"turn {i+1}": {"prompt": prompt, "response": res}}
            )
        
        return res
    
    def multi_turn_response_ollama(self, raw_input: str, style_name: str, prompt_list: list[PromptTemplate])->str:
        input_var = {'input': raw_input}
        self.full_process[style_name] = []
        length = len(prompt_list)
        for i, temp in enumerate(prompt_list):
            prompt = temp.format(**input_var)
            if i < length-1:
                res = self.single_response_ollama(prompt)
            else:
                res = self.single_response_ollama(prompt, 77)
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
    
    def get_summary(self, mul_turn_res: dict):
        summary = self.summary.format(**mul_turn_res)
        if self.api_in_use == 'ollama':
            res = self.single_response_ollama(summary)
        elif self.api_in_use == 'openai':
            res = self.single_response_openai(summary)
        return res, summary
    

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
        # print(f"元素[{self.style}]结束", self.name)
        pass
        
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
        # print(f"元素[{self.style}]结束", self.name)
        pass



if __name__=="__main__":
    print('prompt generator')