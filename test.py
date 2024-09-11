from mprompt_sd import MPromptSD
import torch
import cv2

pipe = MPromptSD(
    model_name="/home/roo/dream/wutr/SD1.4/stable-diffusion-v1-4",
    guidance_scale=10.0,
    scheduler_name="unipc",
)

pipe.to('cuda')

prompt = 'green apple'
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
prompt_list = [prompt]*1 + ['yellow pear']*8

# tensor1, tensor2 = pipe.prompt_to_embedding(prompt, negative_prompt)
# print(tensor1.shape, tensor2.shape)
# # print(tensor1)

# tensor1, tensor2 = pipe.prompt_to_embedding(prompt_list, negative_prompt)
# print(tensor1.shape, tensor2.shape)
# # cnt = 0
# # for i in range(0, 256, 256):
# #     print(tensor1[i:i+4].shape, tensor2.shape)
# #     cnt+=1
# # print(cnt)

# latent = pipe.prepare_latents(256)
# print(latent.shape)
# print(latent)
imgs = pipe.mprompt_generate(prompt_list, num_per_group=3, steps_list=[3, 10, 10])
# print(imgs.shape)

for i in range(imgs.shape[0]):
    cv2.imwrite(f'test_{i}.png', cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))


# a = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
# print(a.reshape(2, 4))