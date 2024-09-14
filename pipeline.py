import os
import yaml
import argparse
import json
from mprompt_sd import MPromptSD, MultiProcessSD


def get_prompt_list(path: str, 
                    step_describe: list[str]
                    )->tuple[list, list]:
    prompt_list = []
    raw_prompt_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_prompt_list.append(data['raw_prompt'])
            for i in step_describe:
                prompt_list.append(data[i])

    return prompt_list, raw_prompt_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SD Config')
    parser.add_argument("--cfg_path", default="./config/sd_cfg.yml")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--guidance_scale", default=None)
    parser.add_argument("--scheduler_name", default=None)
    parser.add_argument("--frozen", default=None)
    parser.add_argument("--num_per_group", default=None)
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--step_list", default=None)
    parser.add_argument("--num_inference_steps", default=None)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--save_path", default=None)
    
    args = parser.parse_args()
    cfg_path = args.__dict__['cfg_path']
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    
    args = parser.parse_args()
    sd_args = args.__dict__['sd_config']
    for k in sd_args.keys():
        if args.__dict__[k] is not None:
            sd_args[k] = args.__dict__[k]
    
    prompt_path = args.__dict__["prompt_path"]
    step_describe = args.__dict__["step_describe"]
    plist, rlist = get_prompt_list(prompt_path, step_describe)
    
    pipe = MultiProcessSD()
    # rp_list = []
    # for i in rlist:
    #     rp_list += [i]*4
    pipe.run(plist, rlist, **sd_args)
    
    
    