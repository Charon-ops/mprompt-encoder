from typing import Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    SchedulerMixin,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


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
        generator: torch.Generator = torch.cuda.manual_seed(1002), 
        torch_device: str = "cuda"
    ) -> torch.FloatTensor:
        """
        Generates a random latent tensor.

        Args:
            generator (Optional[torch.Generator], optional): Generator for random number generation. Defaults to None.
            torch_device (str, optional): Device to store the tensor. Defaults to "cpu".

        Returns:
            torch.FloatTensor: Random latent tensor.
        """
        channel = self.unet.config.in_channels
        height = self.unet.config.sample_size
        width = self.unet.config.sample_size
        latent = torch.randn(
            (batch_size, channel, height, width),
            generator=generator,
            device=torch_device,
        )
        
        return latent
    
    def mprompt_generate(
        self,
        prompt: Union[str, list[str]],
        num_per_group: int = 4,
        negative_prompt: str = "",
        latents: Optional[torch.FloatTensor] = None,
        steps_list: list[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> np.ndarray:
        # check prompt group
        if isinstance(prompt, list):
            assert len(prompt) % num_per_group == 0, "Error in split prompt group"
        
        # Prepare time steps
        if steps_list is None:
            num = num_inference_steps
            num_p = num_per_group
            steps_list = [num//num_p]*(num_p-1) + [num-num//num_p*(num_p-1)]
        else:
            assert len(steps_list) == num_per_group, "Error in split prompt group"
            num_inference_steps = sum(steps_list)
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
        print(f"generate {batch_size} images, inference steps {num_inference_steps}")
        for t in tqdm(self.scheduler.timesteps):
        # for t in self.scheduler.timesteps:
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
            if i == steps_list[s]:
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
    
    