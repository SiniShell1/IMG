import torch
from PIL import Image
import json
import gradio as gr
import os

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from src.run_llava_gen import eval_model
from src.resampler import Resampler, Resampler_cross
from src.utils import get_ip_feat, get_random_instruction

from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from third_party.ip_adapter import IPAdapterPlusXL_fp16 as IPAdapterPlusXL

with open("assets/edit_instructions.json", 'r') as file:
    instructions = json.load(file)
    
weight_dtype = torch.float16
lm_device = "cuda:0"
sdxl_device = "cuda:1"

model_path = "liuhaotian/llava-v1.5-13b"
lm_tokenizer, lm_model, lm_image_processor, lm_context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device_map=lm_device
)
lm_model = lm_model.to(lm_device, dtype=torch.bfloat16)
lm_model.eval()

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
dpo_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
dpo_pipe = StableDiffusionXLPipeline.from_pretrained(dpo_model_id, torch_dtype=weight_dtype, variant="fp16", use_safetensors=True).to(sdxl_device)

dpo_unet_id = "mhdang/dpo-sdxl-text2image-v1"
dpo_unet = UNet2DConditionModel.from_pretrained(dpo_unet_id, subfolder="unet", torch_dtype=weight_dtype)
dpo_pipe.unet = dpo_unet
dpo_pipe = dpo_pipe.to(sdxl_device)

ip_image_processor = CLIPImageProcessor()

ip_resampler = Resampler(dim=1280, depth=4, dim_head=64, heads=20, num_queries=16, embedding_dim=1280, output_dim=2048, ff_mult=4,)
lm_resampler  = Resampler_cross(dim=2048, depth=4, dim_head=64, heads=20, num_queries=16, embedding_dim=5120, output_dim=2048, ff_mult=4,)
                
ip_clip = CLIPVisionModel.from_pretrained("ckpts/image_encoder")
ip_resampler.load_state_dict(torch.load("ckpts/resampler.pth"))

ip_resampler.requires_grad_(False)
ip_clip.requires_grad_(False)

ip_resampler = ip_resampler.to(sdxl_device, dtype=weight_dtype)
ip_clip = ip_clip.to(sdxl_device, dtype=weight_dtype)
lm_resampler = lm_resampler.to(sdxl_device, dtype=torch.float32)

ip_resampler.eval()
ip_clip.eval()
lm_resampler.eval()

stable_diffusion_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(stable_diffusion_model_path, torch_dtype=torch.float16, variant="fp16", add_watermarker=False, use_safetensors=True).to(sdxl_device)
ip_model = IPAdapterPlusXL(pipe, "ckpts/image_encoder", "ckpts/ip-adapter-plus_sdxl_vit-h.bin", sdxl_device, num_tokens=16)

model_ckpt = "ckpts/sdxl_resampler2-90000.pth"
model_state_dict = torch.load(model_ckpt, map_location='cpu')
new_state_dict = {}
for k, v in model_state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  
    else:
        new_state_dict[k] = v
lm_resampler.load_state_dict(new_state_dict)


def generate_images(text_prompt, seed, sdxl_org_guidance_scale, sdxl_ip_guidance_scale):
    sdxl_outer_loop_num = 1
    sdxl_inner_loop_num = 3
    sdxl_interpolation_coeff = 0.0
    sdxl_image_embed_scale = 0.2
    sdxl_layer = -1

    dpo_image = dpo_pipe(text_prompt,generator=torch.Generator().manual_seed(seed), num_inference_steps=50, guidance_scale=sdxl_org_guidance_scale).images[0]  
    all_images = []

    for i, begin_image in enumerate([dpo_image]):
        collect_list = [begin_image]
        loop_image = begin_image
        current_ip_model = ip_model    
        for _ in range(sdxl_outer_loop_num):
            loop_ip_feat = get_ip_feat(loop_image, ip_image_processor, ip_clip, ip_resampler)
            random_instruction = get_random_instruction(instructions)
            prompt = random_instruction.replace("{prompt}", text_prompt)
            lm_args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": prompt,
                "conv_mode": None,
                "image_file": [loop_image],
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()

            with torch.no_grad():
                hidden_states, _ = eval_model(lm_args, lm_tokenizer, lm_model, lm_image_processor, lm_context_len, dtype=torch.bfloat16, generate=False, layer=sdxl_layer)
                hidden_states = hidden_states.to(dtype=torch.float32, device=sdxl_device)
                inner_loop_state = loop_ip_feat.to(dtype=torch.float32, device=sdxl_device)
                for loop_time in range(sdxl_inner_loop_num):
                    inner_loop_state = lm_resampler(hidden_states, inner_loop_state)
        
            modified_bad_image_features = (1-sdxl_interpolation_coeff) * inner_loop_state + sdxl_interpolation_coeff * loop_ip_feat
            modified_bad_image_features = modified_bad_image_features.to(device=sdxl_device, dtype=weight_dtype)

            generated_image = current_ip_model.generate_from_feat(feat=modified_bad_image_features, num_samples=1, num_inference_steps=50, scale=sdxl_image_embed_scale, prompt=text_prompt, seed=seed, guidance_scale=sdxl_ip_guidance_scale)        
            collect_list.append(generated_image[0])
            loop_image = generated_image[0]
            
        images_per_row = 10
        num_rows = (len(collect_list) + images_per_row - 1) // images_per_row
        single_width = collect_list[0].width
        single_height = collect_list[0].height
        total_width = single_width * min(len(collect_list), images_per_row)
        total_height = single_height * num_rows
        stitched_image = Image.new('RGB', (total_width, total_height))
        for i, img in enumerate(collect_list):
            row = i // images_per_row
            col = i % images_per_row
            stitched_image.paste(img, (col * single_width, row * single_height))
        all_images.append(stitched_image)

    total_width = all_images[0].width
    total_height = all_images[0].height * len(all_images)
    sdxl_image = Image.new('RGB', (total_width, total_height))
    for i, img in enumerate(all_images):
        sdxl_image.paste(img, (0, i * all_images[0].height))

    return sdxl_image

iface = gr.Interface(
    fn=generate_images,  
    inputs=[gr.Textbox(label="Enter your prompt"), 
    gr.Number(label="Seed", value=306),
    gr.Slider(label="Original Guidance Scale", minimum=1, maximum=10, step=0.1, value=7.5),
    gr.Slider(label="IP Guidance Scale", minimum=1, maximum=10, step=0.1, value=7.5)], 
    outputs=gr.Image(label="Generated Image 1", type="pil")  
)

iface.launch()