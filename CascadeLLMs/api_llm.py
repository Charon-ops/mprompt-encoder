import openai
from openai import OpenAI
import threading
import yaml
from langchain.prompts import PromptTemplate
import time

        
class PromptGenerator:
    def __init__(self, 
                 api_set: dict, 
                 style_template: list[dict],
                 sys_ins: str):
        
        self.sys_ins = sys_ins
        self.threads = []
        self.Lock = threading.Lock()
        
        self.result_dict: dict[str, str] = {}
        self.full_process: dict[str, list] = {}
        
        self.init_openai_api(api_set)
        self.load_style_template(style_template)
        pass
    
    def init_openai_api(self, api_set: dict):
        self.api_key = api_set['api_key']
        self.base_url = api_set['base_url']
        self.model = api_set['model']
        openai._client = OpenAI(
            api_key=self.api_key, # 替换成真实的API_KEY
            base_url=self.base_url, # 服务endpoint
        )
        pass
    
    def load_style_template(self, style_template: list[dict]):
        self.prompt_template_dict = {}
        for style in style_template:
            self.prompt_template_dict[style['name']] = style['template']
        pass
    
    def single_response(self, prompt: str)->str:
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
    
    def multi_turn_response(self, raw_input: str, style_name: str, prompt_list: list[PromptTemplate])->str:
        input_var = {'input': raw_input}
        self.full_process[style_name] = []
        
        for i, temp in enumerate(prompt_list):
            prompt = temp.format(**input_var)
            res = self.single_response(prompt)
            input_var = {'input': res}
            
            self.full_process[style_name].append(
                {f"turn {i+1}": {"prompt": prompt, "response": res}}
            )
        
        return res
    
    def multi_thread_response(self, raw_input: str)->dict:
        self.result_dict = {}
        self.full_process = {}
        
        for style_name, temp_list in self.prompt_template_dict.items():
            tmp_thread = ResponseThread(style_name, raw_input, temp_list, self)
            tmp_thread.start()
            self.threads.append(tmp_thread)
            
        for t in self.threads:
            t.join()
        
        return self.result_dict, self.full_process
        

class ResponseThread(threading.Thread):
    def __init__(self, style: str, raw_input: str, prompt_list: list, 
                 generator: PromptGenerator):
        threading.Thread.__init__(self)
        self.style = style
        self.raw_input = raw_input
        self.prompt_list = prompt_list
        self.generator = generator
 
    def run(self):
        print(f"风格[{self.style}]开始", self.name)
        res = self.generator.multi_turn_response(self.raw_input, self.style, self.prompt_list)
        self.generator.Lock.acquire()
        self.generator.result_dict[self.style] = res
        self.generator.Lock.release()
        
    def __del__(self):
        print(f"风格[{self.style}]结束", self.name)



if __name__=="__main__":
    
    with open("/home/roo/dream/wutr/mprompt-encoder/config/templates.yml", 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    styles = data['styles']
    api_set = data['api_set']
    sys_ins = data['sys_ins']
    gener = PromptGenerator(api_set, styles, sys_ins)
    res, his = gener.multi_thread_response("一个红色的苹果")
    print(res.keys())
    print(his)