import openai
from openai import OpenAI
import threading
from langchain.prompts import PromptTemplate
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
num_style = 1 # 风格数，每个风格单独一个线程
style_list = ['1']
prompt_template_dict = {} # prompt
# system_instruction_list = ["重复用户给你的内容，不要改变任何内容，记住一定只能重复。"]
system_instruction_list = ["你是一个AI助手，你要完成用户给你的所有指令，不允许有差错"]

# 模板
style1_turn1 = '''
你是一个专门的艺术风格识别专家。你的任务是分析用户提供的图像描述或风格需求，并识别出基本的艺术风格特征。请遵循以下步骤：

仔细阅读用户输入，提取关键词和概念。
基于这些信息，识别主要的艺术风格（如印象派、抽象主义、浪漫主义等）。
列出该风格的3-5个最显著的基本特征。
提供一个简短的总结，概括这个风格的整体感觉和氛围。

请以简洁、专业的方式呈现你的分析结果。你的输出将作为下一阶段风格细节分析的基础。

用户输入：{input}
'''
style1_turn2 = '''
你是一个精通艺术风格细节的专家。你的任务是基于前一阶段识别出的基本风格特征，进一步扩充和详细描述具体的风格元素。同时也希望你完善前一阶段的不足。请按照以下步骤进行：

仔细阅读LLM1提供的风格识别结果。
对于每个提到的基本特征，提供2-3个更具体的表现形式或技巧。
描述这个风格常用的色彩方案、笔触技巧、构图原则等。
列举2-3位代表性艺术家的作品特点。
提供一些可能在这种风格中出现的具体视觉元素或主题。

请以结构化的方式呈现你的分析，使其易于理解和转化为具体的图像描述。你的输出将用于下一阶段的风格语言转换。

LLM1的输出：{input}
'''
style1_turn3 = '''
你是一个专门的文生图提示词专家。你的任务是将前两个阶段提供的艺术风格特征和细节转化为适合文生图模型使用的描述语言。同时也希望你完善前一阶段的不足。请遵循以下指南：

仔细阅读LLM2提供的风格分析和细节。
将这些信息转化为简洁、具体且富有描述性的短语。
使用文生图模型常用的关键词和格式，如特定的标点符号、权重标记等。
包含颜色、构图、光影、质地等方面的详细描述。
添加一些技术性词汇，如渲染方式、艺术媒介等。
确保使用的语言和术语是文生图模型能够理解和处理的。

你的输出应该是一系列强大、精确的提示词，这些词能够指导文生图模型生成符合特定艺术风格的图像。

LLM2的输出：{input}
'''
style1_turn4 = '''
你是一个专门负责确保艺术风格描述一致性和连贯性的专家。你的任务是审查并优化前一阶段生成的文生图提示词，确保它们能够产生风格统一、协调的图像。请按照以下步骤进行：

仔细阅读LLM3提供的文生图提示词。
检查所有元素是否与最初识别的艺术风格保持一致。
确保各个描述之间没有矛盾或冲突。
调整词语顺序，使整体描述更加流畅和有逻辑。
如果需要，添加转换词或短语来增强连贯性。
确保关键风格元素得到适当的强调和权重。
移除任何可能导致风格偏离的不必要或矛盾的描述。
最后，提供一个简洁的总结，概括整体风格和预期的视觉效果。

你的输出应该是一个经过优化、高度连贯且风格一致的文生图提示词集合，可以直接用于生成符合预期艺术风格的图像。

LLM3的输出：{input}
'''

style1_prompt = []
style1_prompt.append(style1_turn1)
style1_prompt.append(style1_turn2)
style1_prompt.append(style1_turn3)
style1_prompt.append(style1_turn4)




# tmp_prompt = "重复这个句子，不要改变或添加任何内容，编写一首诗，解释编程中的递归概念。"
tmp_prompt = "描绘一只猫"

# 全局变量
Lock = threading.Lock()
threads = []
result_dict = {}

def get_prompt_template_dict()->None:
    prompt_template_list = []
    for temp in style1_prompt:
        prompt_template_list.append(
            PromptTemplate(
                input_variables=['input'],
                template=temp
            )
        )
    prompt_template_dict['1'] = prompt_template_list


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

def multi_turn_response(raw_input: str, prompt_list: list[PromptTemplate])->str:
    input_var = {'input': raw_input}
    for i, temp in enumerate(prompt_list):
        prompt = temp.format(**input_var)
        res = single_response(prompt)
        input_var = {'input': res}
        print(f"turn {i}")
        print('prompt: \n'+prompt)
        print('result: \n'+res)
        print('\n')
    
    return res

def multi_thread_response(raw_input: str)->None:
    global result_dict
    result_dict = {}
    
    for s in style_list:
        prompt_list = prompt_template_dict[s]
        tmp_thread = responseThread(s, raw_input, prompt_list)
        tmp_thread.start()
        threads.append(tmp_thread)
        
    for t in threads:
        t.join()

class responseThread(threading.Thread):
    def __init__(self, style: str, raw_input: str, prompt_list: list):
        threading.Thread.__init__(self)
        self.style = style
        self.raw_input = raw_input
        self.prompt_list = prompt_list
 
    def run(self):
        print(f"风格[{self.style}]开始", self.name)
        res = multi_turn_response(self.raw_input, self.prompt_list)
        Lock.acquire()
        result_dict[self.style] = res
        Lock.release()
        
    def __del__(self):
        print(f"风格[{self.style}]结束", self.name)



if __name__=="__main__":
    get_prompt_template_dict()
    multi_thread_response("一个红色的苹果")
    print(result_dict)
    print(result_dict['1'])
    # print(style1_turn1)
    pass