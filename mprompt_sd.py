from typing import Optional, Union
import os
import math
import time
import numpy as np
import torch
import cv2
import multiprocessing
from multiprocessing import RLock, Queue, Process
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    SchedulerMixin,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


lock = RLock()
multiprocessing.set_start_method('spawn', force=True)

class MPromptSD:
    def __init__(self,
                 model_name: str,
                 scheduler_name: str = "unipc",
                 frozen: bool = True,
                 guidance_scale: float = 7.5,
                 ):
        # Initialize generator
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae", 
            use_safetensors=True
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, 
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            use_safetensors=True
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name, 
            subfolder="unet", 
            use_safetensors=True
        )
        
         # Initialize scheduler
        if scheduler_name == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(
                model_name, 
                subfolder="scheduler"
            )
        elif scheduler_name == "unipc":
            self.scheduler = UniPCMultistepScheduler.from_pretrained(
                model_name, 
                subfolder="scheduler"
            )
        else:
            raise ValueError(
                "Invalid scheduler name (ddim, unipc) and not specify scheduler."
            )
            
        # Initialize guidance scale
        self.guidance_scale = guidance_scale
        
        # Initialize parameters
        if frozen:
            for param in self.unet.parameters():
                param.requires_grad = False

            for param in self.text_encoder.parameters():
                param.requires_grad = False

            for param in self.vae.parameters():
                param.requires_grad = False
        
    def to(self, *args, **kwargs):
        # to cuda
        self.vae.to(*args, **kwargs)
        self.text_encoder.to(*args, **kwargs)
        self.unet.to(*args, **kwargs)
        self.torch_device = self.unet.device
    
    @torch.no_grad()
    def prompt_to_embedding(
        self, prompt: Union[str, list[str]], negative_prompt: str = ""
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if isinstance(prompt, str):
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        elif isinstance(prompt, list):
            text_input = self.tokenizer.batch_encode_plus(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        # print(text_input.input_ids.shape)
        # b, 77

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]
        # print(text_embeddings.shape)
        # b, 77, 768
        
        negative_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_embeddings = self.text_encoder(
            negative_input.input_ids.to(self.torch_device)
        )[0]

        return text_embeddings, negative_embeddings
    
    def prepare_latents(
        self,
        batch_size: int,
        generator: torch.Generator = torch.cuda.manual_seed(42), 
    ) -> torch.FloatTensor:
        channel = self.unet.config.in_channels
        height = self.unet.config.sample_size
        width = self.unet.config.sample_size
        latent = torch.randn(
            (batch_size, channel, height, width),
            generator=generator,
            device=self.torch_device,
        )
        
        return latent
    
    def mprompt_generate(
        self,
        prompt: Union[str, list[str]],
        num_per_group: int = 4,
        negative_prompt: str = "",
        latents: Optional[torch.FloatTensor] = None,
        step_list: list[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> np.ndarray:
        # check prompt group
        if isinstance(prompt, list):
            assert len(prompt) % num_per_group == 0, "Error in split prompt group"
        
        # Prepare time steps
        if step_list is None:
            num = num_inference_steps
            num_p = num_per_group
            step_list = [num//num_p]*(num_p-1) + [num-num//num_p*(num_p-1)]
        else:
            assert len(step_list) == num_per_group, "Error in split prompt group"
            num_inference_steps = sum(step_list)
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Encode input prompt
        emb, negative_emb = self.prompt_to_embedding(prompt, negative_prompt)
        
        # Set parameters
        if isinstance(prompt, str):
            batch_size = 1
            emb = torch.cat([emb]*num_per_group).unsqueeze(0)
        elif isinstance(prompt, list):
            batch_size = len(prompt) // num_per_group
            emb = emb.reshape(batch_size, num_per_group, 77, 768)
            negative_emb = torch.cat([negative_emb]*batch_size)
        
        # Prepare latents
        if latents is None:
            latents = self.prepare_latents(batch_size)
        
        # Denoising loop
        i = 0
        s = 0 # stages cnt
        stage_emb = emb[:, s, :, :]
        # print("stage ", stage_emb.shape)
        # print(f"generate {batch_size} images, inference steps {num_inference_steps}")
        # for t in tqdm(self.scheduler.timesteps):
        for t in self.scheduler.timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, timestep=t)
            with torch.no_grad():
                # Predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, 
                    encoder_hidden_states=stage_emb
                ).sample
                attn_proc = AttnProcessor2_0()
                self.unet.set_attn_processor(processor=attn_proc)
                noise_uncond = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=negative_emb
                ).sample
            # perform guidance
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            i += 1
            if i == step_list[s]:
                i = 0
                s += 1
                if s >= num_per_group:
                    continue
                stage_emb = emb[:, s, :, :]
        # Decode the images
        # print(self.vae.config.scaling_factor) 0.18215
        latents = 1 / self.vae.config.scaling_factor * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        images = (image / 2 + 0.5).clamp(0, 1)
        images = (images.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        return images
    
    
class MultiProcessSD:
    def __init__(self):
        assert torch.cuda.is_available(), "No CUDA-enabled device found."
        self.cuda_cnt = torch.cuda.device_count()
    
    def split_prompt_list(
        self, 
        prompt_list: list[str],
        num_per_group: int,
        raw_prompt_list: list[str]
    ) -> list[list]:
        # check
        assert isinstance(prompt_list, list), "prompt_list must be list[str]"
        assert len(prompt_list) % num_per_group == 0, "Error in split prompt group"
        assert len(raw_prompt_list) * num_per_group == len(prompt_list), "Error in raw_prompt_list"
        
        num_group = len(prompt_list) // num_per_group
        num_per_cuda = num_group // self.cuda_cnt
        prompts: list[list] = []
        raw_prompts: list[list] = []
        for i in range(0, self.cuda_cnt):
            idx = i*num_per_cuda*num_per_group
            prompts.append(
                prompt_list[idx: idx+num_per_cuda*num_per_group]
            )
            raw_prompts.append(
                raw_prompt_list[i*num_per_cuda: (i+1)*num_per_cuda]
            )
            
        res = num_group % self.cuda_cnt
        if res > 0:
            res = num_group - res
            # 多的尽量均匀分
            for i, j in enumerate(range(res, num_group)):
                idx = j*num_per_group
                prompts[i] += prompt_list[idx: idx+num_per_group]
                raw_prompts[i] += raw_prompt_list[j: j+1]
                    
        return prompts, raw_prompts
        
    def run(
        self, 
        prompt_list: list[str],
        raw_prompt_list: list[str],
        **kwargs,
    ):
        prompts, raw_prompts = self.split_prompt_list(
            prompt_list,
            kwargs['num_per_group'],
            raw_prompt_list,
        )
        
        print(f'父进程 {os.getpid()}')
        processes = []
        
        for idx in range(self.cuda_cnt):
            kwds = kwargs
            kwds['prompt'] = prompts[idx]
            kwds['raw_prompt'] = raw_prompts[idx]
            kwds['cuda_index'] = idx
            p = Process(target=sd_process_sub, args=(kwds,))
            p.start()
            processes.append(p)
        
        while True:
            if all(p.exitcode is not None for p in processes):
                break
            
        # 确保所有进程完成
        for p in processes:
            p.join()
        
def sd_process_sub(arg_dict: dict):
    sd_process(**arg_dict)
    
        
def sd_process(
    prompt: list[str],
    num_per_group: int = 4,
    negative_prompt: str = "",
    raw_prompt: list[str] = [""],
    latents: Optional[torch.FloatTensor] = None,
    step_list: Optional[list[int]] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    batch_size: int = 1,
    cuda_index: int = 0,
    save_path: str = "./",
    model_name: str = "./",
    scheduler_name: str = "unipc",
    frozen: bool = True,
):
    num_group = len(prompt) // num_per_group
    num_batch = num_group // batch_size
    res_group = num_group % batch_size
    prompt_list: list[list[str]] = []
    for i in range(num_batch):
        idx = i*num_per_group*batch_size
        prompt_list.append(
            prompt[idx: idx+num_per_group*batch_size]
        )
    if res_group > 0:
        # 多的单独一个batch
        prompt_list.append(prompt[-res_group*num_per_group:])
        
    pro_bar = tqdm(prompt_list, ncols=100, 
                    desc=f"CUDA {cuda_index} pid {os.getpid()}",
                    delay=0.1,
                    position=cuda_index, 
                    ascii=False
                    )
    lock.acquire()
    sd_model = MPromptSD(model_name,
                         scheduler_name,
                         frozen,
                         guidance_scale,
                        )
    lock.release()
    
    device = torch.device(f'cuda:{cuda_index}')
    sd_model.to(device)
    
    print(f"Starting CUDA {cuda_index}")

    img_idx = 0
    for prompts in pro_bar:
        imgs = sd_model.mprompt_generate(prompts,
                                    num_per_group=num_per_group,
                                    negative_prompt=negative_prompt,
                                    latents=latents,
                                    step_list=step_list,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    )
        for i, j in enumerate(range(img_idx, img_idx+imgs.shape[0])):
            img_name = raw_prompt[j].replace("\n", "")
            cv2.imwrite(os.path.join(save_path, f'{img_name}.png'),
                        cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
        img_idx += imgs.shape[0]
    pro_bar.close()
    