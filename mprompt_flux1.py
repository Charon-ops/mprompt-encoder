import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import os
import numpy as np
import torch
import multiprocessing
from multiprocessing import RLock, Queue, Process
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers import (
    AutoencoderKL,
    FluxTransformer2DModel
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput


lock = RLock()
multiprocessing.set_start_method('spawn', force=True)

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class MPromptFlux:
    def __init__(self,
                 model_name: str,
                 scheduler_name: str = "FlowMatchEuler",
                 frozen: bool = True,
                 guidance_scale: float = 7.5,
                 ):
        # Initialize
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae", 
            use_safetensors=True
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            use_safetensors=True
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, 
            subfolder="tokenizer"
        )
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            model_name,
            subfolder="text_encoder_2",
            use_safetensors=True
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            model_name, 
            subfolder="tokenizer_2"
        )
        self.transformer = FluxTransformer2DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            use_safetensors=True
        )

        if scheduler_name == "FlowMatchEuler":
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_name, 
                subfolder="scheduler"
            )
        else :
            assert False, "Unsupported scheduler name"
        
        # Freeze parameters
        if frozen:
            for param in self.vae.parameters():
                param.requires_grad = False

            for param in self.text_encoder.parameters():
                param.requires_grad = False

            for param in self.text_encoder_2.parameters():
                param.requires_grad = False

            for param in self.transformer.parameters():
                param.requires_grad = False

        # Other parameters
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 64
        self.guidance_scale = guidance_scale
    
    def to(self, *args, **kwargs):
        # to cuda
        self.vae.to(*args, **kwargs)
        self.text_encoder.to(*args, **kwargs)
        self.text_encoder_2.to(*args, **kwargs)
        self.transformer.to(*args, **kwargs)
        self.torch_device = self.transformer.device

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
    ):
        device = self.torch_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        
        return prompt_embeds
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 512,
    ):
        device = self.torch_device
        dtype = self.text_encoder_2.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        device = self.torch_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        assert isinstance(prompt, List[str]), "prompt must be a single string or a list of strings"
        batch_size = len(prompt)
        
        prompt_2 = prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # We only use the pooled prompt output from the CLIPTextModel
        pooled_prompt_embeds = self._get_clip_prompt_embeds(
            prompt=prompt,
        )
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt_2,
            max_sequence_length=max_sequence_length,
        )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        # text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, pooled_prompt_embeds, text_ids
    
    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)
    
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents
    
    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents, latent_image_ids
    
    def mprompt_generate(
        self,
        prompt: Union[str, List[str]] = None,
        num_per_group: int = 4,
        latents: Optional[torch.FloatTensor] = None,
        step_list: list[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        if isinstance(prompt, list):
            assert len(prompt) % num_per_group == 0, "Error in split prompt group"

        if step_list is None:
            num = num_inference_steps
            num_p = num_per_group
            step_list = [num//num_p]*(num_p-1) + [num-num//num_p*(num_p-1)]
        else:
            assert len(step_list) == num_per_group, "Error in split prompt group"
            num_inference_steps = sum(step_list)
        # self.check_inputs(
        #     prompt,
        #     prompt_2,
        #     height,
        #     width,
        #     prompt_embeds=prompt_embeds,
        #     pooled_prompt_embeds=pooled_prompt_embeds,
        #     callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        #     max_sequence_length=max_sequence_length,
        # )

        # self._guidance_scale = guidance_scale
        # self._joint_attention_kwargs = joint_attention_kwargs
        # self._interrupt = False

        # 2. Define call parameters
        # if prompt is not None and isinstance(prompt, str):
        #     batch_size = 1
        # elif prompt is not None and isinstance(prompt, list):
        #     batch_size = len(prompt)

        device = self.torch_device

        lora_scale = None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # Set parameters
        if isinstance(prompt, list):
            batch_size = len(prompt) // num_per_group
            pooled_emb = pooled_prompt_embeds.reshape(batch_size, num_per_group, 77, 768)
            emb = prompt_embeds.reshape(batch_size, num_per_group, max_sequence_length, 768)
            ####################
            print(pooled_emb.shape)
            print(emb.shape)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            None,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        i = 0
        s = 0 # stages cnt
        stage_emb = emb[:, s, :, :]
        stage_pooled_emb = pooled_emb[:, s, :, :]
        for t in timesteps:
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None
            with torch.no_grad():
                noise_pred = self.transformer(
                    hidden_states=latents,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=stage_pooled_emb,
                    encoder_hidden_states=stage_emb,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # if latents.dtype != latents_dtype:
            #     if torch.backends.mps.is_available():
            #         # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
            #         latents = latents.to(latents_dtype)
            i += 1
            if i == step_list[s]:
                i = 0
                s += 1
                if s >= num_per_group:
                    continue
                stage_emb = emb[:, s, :, :]
                stage_pooled_emb = pooled_emb[:, s, :, :]

            

            # if callback_on_step_end is not None:
            #     callback_kwargs = {}
            #     for k in callback_on_step_end_tensor_inputs:
            #         callback_kwargs[k] = locals()[k]
            #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

            #     latents = callback_outputs.pop("latents", latents)
            #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            #     progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


class MultiProcessFlux:
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
    scheduler_name: str = "FlowMatchEuler",
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
    sd_model = MPromptFlux(model_name,
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
                                    latents=latents,
                                    step_list=step_list,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    height=256,
                                    width=256,
                                    max_sequence_length=512,
                                    generator=torch.Generator("cpu").manual_seed(0),
                                    )
        for i, j in enumerate(range(img_idx, img_idx+imgs.shape[0])):
            img_name = raw_prompt[j].replace("\n", "").replace(".", "")
            # cv2.imwrite(os.path.join(save_path, f'{img_name}.png'),
            #             cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            imgs[i].save(os.path.join(save_path, f'{img_name}.png'))
        img_idx += len(imgs)
    pro_bar.close()
    