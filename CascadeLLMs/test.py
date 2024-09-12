import openai
from openai import OpenAI
import ollama
import threading
import yaml
from langchain.prompts import PromptTemplate
import time
from prompt_generator import PromptGenerator




if __name__=="__main__":
    t1 = time.time()
    with open("/home/roo/dream/wutr/mprompt-encoder/config/templates_en.yml", 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    styles = data['styles']
    summary = data['summary']
    api_in_use = data['api_in_use']
    ollama_api_set = data['ollama_api_set']
    openai_api_set = data['openai_api_set']
    sys_ins = data['sys_ins']
    gener = PromptGenerator(api_in_use, openai_api_set, ollama_api_set, styles, summary, sys_ins)
    # summary = PromptTemplate(
    #         input_variables=['artistic_style', 'composition', 'visual_center', 'color_tone'],
    #         template=summary
    #     )
    # vars = {'artistic_style':1, 'composition':2, 'visual_center':3, 'color_tone':4}
    # summary = summary.format(**vars)
    # print(summary)
    res, his = gener.multi_thread_response("一个红色的苹果")
    summ = gener.get_summary(res) # summary
    print(summ)
    
    # print(res.keys())
    # print(his)
    # print("最终结果\n")
    # for k, v in res.items():
    #     print(f"元素: {k}")
    #     print(v)
    #     print('\n')
    
    # print("中间过程\n")
    # for k, v in his.items():
    #     print(f"元素: {k}")
    #     for i, s in enumerate(v):
    #         print(f"turn {i+1}:")
    #         print(f"prompt:\n{s[f'turn {i+1}']['prompt']}")
    #         print(f"response:\n{s[f'turn {i+1}']['response']}")
    #     print('\n')
        
    print(time.time()-t1)