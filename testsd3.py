import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("F:\\#MODELS\\FLUX1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=256,
    width=256,
    guidance_scale=3.5,
    num_inference_steps=10,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")
