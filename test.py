
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path

# lm_device = "cuda:0"
# model_path = "liuhaotian/llava-v1.5-13b"
# lm_tokenizer, lm_model, lm_image_processor, lm_context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path),
#     device_map=lm_device
# )
# import torch
# import os
# from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
# weight_dtype = torch.float16
# lm_device = "cuda:0"
# sdxl_device = "cuda:1"
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# dpo_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# dpo_pipe = StableDiffusionXLPipeline.from_pretrained(dpo_model_id, torch_dtype=weight_dtype, variant="fp16", use_safetensors=True).to(sdxl_device)

# dpo_unet_id = "mhdang/dpo-sdxl-text2image-v1"
# dpo_unet = UNet2DConditionModel.from_pretrained(dpo_unet_id, subfolder="unet", torch_dtype=weight_dtype)
# dpo_pipe.unet = dpo_unet
# dpo_pipe = dpo_pipe.to(sdxl_device)

# import os
# import subprocess
# import time
# import signal
# import sys
# # sys.path.append("..")
# # sys.path.append("/cephfs/yuanty/yanch/LLaVA")
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
# from llava.eval.run_llava_gen import eval_model
# import random
# from datasets import load_dataset
# from PIL import Image
# import requests
# import copy
# import torch
# from io import BytesIO

# import json
# import warnings
# from resampler import Resampler, Resampler_cross
# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection


# from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
# from diffusers import AutoPipelineForImage2Image
# from diffusers import Transformer2DModel, PixArtSigmaPipeline, PixArtAlphaPipeline
# from diffusers import DiffusionPipeline


# from third_party.ip_adapter import IPAdapterPlusXL_fp16 as IPAdapterPlusXL
# import argparse
# import os
# from utils import get_ip_feat, get_random_instruction
# from tqdm import tqdm
# import gradio as gr
# from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
# weight_dtype = torch.float16
# lm_device = "cuda:0"
# sdxl_device = "cuda:1"
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# ip_image_processor = CLIPImageProcessor()

# ip_resampler = Resampler(dim=1280, depth=4, dim_head=64, heads=20, num_queries=16, embedding_dim=1280, output_dim=2048, ff_mult=4,)
# lm_resampler  = Resampler_cross(dim=2048, depth=4, dim_head=64, heads=20, num_queries=16, embedding_dim=5120, output_dim=2048, ff_mult=4,)
                
# ip_clip = CLIPVisionModel.from_pretrained("ckpts/image_encoder")
# ip_resampler.load_state_dict(torch.load("ckpts/resampler.pth"))

# ip_resampler.requires_grad_(False)
# ip_clip.requires_grad_(False)

# ip_resampler = ip_resampler.to(sdxl_device, dtype=weight_dtype)
# ip_clip = ip_clip.to(sdxl_device, dtype=weight_dtype)
# lm_resampler = lm_resampler.to(sdxl_device, dtype=torch.float32)

# ip_resampler.eval()
# ip_clip.eval()
# lm_resampler.eval()

# stable_diffusion_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
# pipe = StableDiffusionXLPipeline.from_pretrained(stable_diffusion_model_path, torch_dtype=torch.float16, variant="fp16", add_watermarker=False, use_safetensors=True).to(sdxl_device)


# ip_model = IPAdapterPlusXL(pipe, "ckpts/image_encoder", "ckpts/ip-adapter-plus_sdxl_vit-h.bin", sdxl_device, num_tokens=16)

# model_ckpt = "ckpts/sdxl_resampler2-90000.pth"
# model_state_dict = torch.load(model_ckpt, map_location='cpu')
# new_state_dict = {}
# for k, v in model_state_dict.items():
#     if k.startswith('module.'):
#         new_state_dict[k[7:]] = v  
#     else:
#         new_state_dict[k] = v
# lm_resampler.load_state_dict(new_state_dict)


# import gradio as gr
# from diffusers import StableDiffusionXLPipeline
# import torch

# weight_dtype = torch.float16
# sdxl_device = "cuda:0"
# dpo_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# pipe = StableDiffusionXLPipeline.from_pretrained(dpo_model_id, torch_dtype=weight_dtype, variant="fp16", use_safetensors=True).to(sdxl_device)

# def generate_images(prompt):
#     print(prompt)
#     image1 = pipe(prompt).images[0]
#     # image2 = pipe(prompt).images[0]
#     image1.save("image1.png")

#     return image1

# iface = gr.Interface(
#     fn=generate_images,  
#     inputs=gr.Textbox(label="Enter your prompt"),  
#     outputs=[gr.Image(label="Generated Image 1", type="pil")]  
# )

# iface.launch()

import os
import subprocess
import time
import signal
import sys
import random
from datasets import load_dataset
from PIL import Image
import requests
import copy
import torch
from io import BytesIO
import sys
import json
import warnings
from resampler import Resampler, Resampler_cross

import torch


import argparse

import os
from utils import get_ip_feat, get_random_instruction

import sys
import os
import argparse
import torch

import numpy
from io import BytesIO
from tqdm import tqdm

from third_party.src.flux.modules.layers import ImageProjModel
from third_party.src.flux.util import load_checkpoint
from third_party.src.flux.xflux_pipeline import XFluxPipeline
import numpy as np
flux_device = "cuda:0"
lm_device = "cuda:1"
args_ip_repo_id = "XLabs-AI/flux-ip-adapter"
args_ip_name = "ip_adapter.safetensors"
args_model_type = "flux-dev"
# args_model_type = "/nas/shared/sys2/chuanhaoyan/workspace/flux"
args_ip_local_path = None
args_offload = False
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
xflux_pipeline = XFluxPipeline(args_model_type, flux_device, args_offload)
print('load ip-adapter:', args_ip_local_path, args_ip_repo_id, args_ip_name)
xflux_pipeline.set_ip(args_ip_local_path, args_ip_repo_id, args_ip_name)

flux_lm_resampler  = Resampler_cross(dim=4096, depth=4, dim_head=64, heads=20, num_queries=16, embedding_dim=5120, output_dim=4096, ff_mult=4,)
flux_lm_resampler = flux_lm_resampler.to(lm_device, dtype=torch.float32)
flux_lm_resampler.eval()

flux_model_ckpt = 'ckpts/flux_resampler2-70000.pth'
flux_model_state_dict = torch.load(flux_model_ckpt, map_location='cpu')
flux_new_state_dict = {}
for k, v in flux_model_state_dict.items():
    if k.startswith('module.'):
        flux_new_state_dict[k[7:]] = v  
    else:
        flux_new_state_dict[k] = v
flux_lm_resampler.load_state_dict(flux_new_state_dict)

args_width = 1024
args_height = 1024
neg_image_prompt = np.zeros((args_width, args_height, 3), dtype=np.uint8)
neg_image_proj = xflux_pipeline.get_image_proj(neg_image_prompt)