import torch
from diffusers import StableDiffusion3Pipeline

path = "/home/roo/dream/wutr/mprompt-encoder/ckpts/SD3"
pipe = StableDiffusion3Pipeline.from_pretrained(path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image
