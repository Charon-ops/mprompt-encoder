from mprompt_sd1_4 import MPromptSD, MultiProcessSD
import torch
import cv2
import numpy as np


# Test split
if __name__ == "__main__":
    plist = [str(i) for i in range(400)]
    rlist = [str(i) for i in range(100)]
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    test = MultiProcessSD()
    set_dict = {"num_per_group": 4,
                "negative_prompt": negative_prompt, 
                "num_inference_steps": 20,
                "guidance_scale": 10,
                "batch_size": 2,
                "save_path": './test_imgs',
                "model_name": '/home/roo/dream/wutr/SD-exp/stable-diffusion-v1-4'
                }
    test.run(plist, rlist, **set_dict)
# print(len(test.split_prompt_list(plist, 4, rlist)[1][0]))
# print(test.sd_process(plist, batch_size=3))

# a1 = np.empty([1, 512, 512, 3])
# a2 = np.random.rand(2, 512, 512, 3)
# a1 = np.concatenate((a1, a2))
# a1 = np.concatenate((a1, a2))
# # np.concatenate((a1, a1))
# print(a1.shape)
# assert 0
# pipe = MPromptSD(
#     model_name="/home/roo/dream/wutr/SD-exp/stable-diffusion-v1-4",
#     guidance_scale=10.0,
#     scheduler_name="unipc",
# )

# pipe.to('cuda')

# prompt = 'green apple'
# negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
# prompt_list = [prompt]*1 + ['yellow pear']*8

# # tensor1, tensor2 = pipe.prompt_to_embedding(prompt, negative_prompt)
# # print(tensor1.shape, tensor2.shape)
# # # print(tensor1)

# # tensor1, tensor2 = pipe.prompt_to_embedding(prompt_list, negative_prompt)
# # print(tensor1.shape, tensor2.shape)
# # # cnt = 0
# # # for i in range(0, 256, 256):
# # #     print(tensor1[i:i+4].shape, tensor2.shape)
# # #     cnt+=1
# # # print(cnt)

# # latent = pipe.prepare_latents(256)
# # print(latent.shape)
# # print(latent)
# imgs = pipe.mprompt_generate(prompt_list, num_per_group=3, steps_list=[3, 10, 10])
# # print(imgs.shape)

# for i in range(imgs.shape[0]):
#     cv2.imwrite(f'test_{i}.png', cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))


# # a = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
# # print(a.reshape(2, 4))