import torch
import json
from PIL import Image
import numpy as np
import gradio as gr

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from third_party.flux.modules.layers import ImageProjModel
from third_party.flux.util import load_checkpoint
from third_party.flux.xflux_pipeline import XFluxPipeline

from src.resampler import Resampler, Resampler_cross
from src.utils import get_ip_feat, get_random_instruction
from src.run_llava_gen import eval_model

with open("assets/edit_instructions.json", 'r') as file:
    instructions = json.load(file)
    
weight_dtype = torch.float16
lm_device = "cuda:0"
flux_device = "cuda:1"


model_path = "liuhaotian/llava-v1.5-13b"
lm_tokenizer, lm_model, lm_image_processor, lm_context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device_map=lm_device
)
lm_model = lm_model.to(lm_device, dtype=torch.bfloat16)
lm_model.eval()


args_ip_repo_id = "XLabs-AI/flux-ip-adapter"
args_ip_name = "ip_adapter.safetensors"
args_model_type = "flux-dev"
args_ip_local_path = None
args_offload = False
xflux_pipeline = XFluxPipeline(args_model_type, flux_device, args_offload)
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


def generate_image(text_prompt, flux_seed, flux_true_gs_org, flux_true_gs_ip, flux_num_steps):
    flux_timestep_to_start_cfg = 1
    flux_control_weight = 0.8
    flux_neg_prompt = ""
    flux_outer_loop_num = 1
    flux_inner_loop_num = 3
    flux_interpolation_coeff  = 0.0
    flux_image_embed_scale = 0.2
    flux_layer = -1 
    org_image = xflux_pipeline.forward(
        prompt=text_prompt,
        width=args_width,
        height=args_height,
        guidance=3.5,
        num_steps=flux_num_steps,
        seed=flux_seed,
        controlnet_image=None,
        timestep_to_start_cfg=flux_timestep_to_start_cfg,
        true_gs=flux_true_gs_org,
        control_weight=flux_control_weight,
        neg_prompt=flux_neg_prompt,
        image_proj=neg_image_proj,
        neg_image_proj=neg_image_proj,
        ip_scale=0.0,
        neg_ip_scale=0.0,
    )

    image_embed_list = []
    all_images = []
    for i, begin_image in enumerate([org_image]):
        collect_list = [begin_image]
        loop_image = begin_image
        
        for _ in range(flux_outer_loop_num):
            loop_ip_feat = xflux_pipeline.get_image_proj(loop_image).to(device=lm_device)

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
                hidden_states, _ = eval_model(lm_args, lm_tokenizer, lm_model, lm_image_processor, lm_context_len, dtype=torch.bfloat16, generate=False, layer=flux_layer)
                hidden_states = hidden_states.to(dtype=torch.float32)
                inner_loop_state = loop_ip_feat.to(dtype=torch.float32)
                for loop_time in range(flux_inner_loop_num):
                    inner_loop_state = flux_lm_resampler(hidden_states, inner_loop_state)
        
            modified_bad_image_features = (1-flux_interpolation_coeff) * inner_loop_state + flux_interpolation_coeff * loop_ip_feat
            modified_bad_image_features = modified_bad_image_features.to(device=flux_device, dtype=torch.bfloat16)

            generated_image = xflux_pipeline.forward(
                prompt=text_prompt,
                width=args_width,
                height=args_height,
                guidance=3.5,
                num_steps=flux_num_steps,
                seed=flux_seed,
                controlnet_image=None,
                timestep_to_start_cfg=flux_timestep_to_start_cfg,
                true_gs=flux_true_gs_ip,
                control_weight=flux_control_weight,
                neg_prompt=flux_neg_prompt,
                image_proj=modified_bad_image_features,
                neg_image_proj=neg_image_proj,
                ip_scale=flux_image_embed_scale,
                neg_ip_scale=flux_image_embed_scale,
            )
            
            collect_list.append(generated_image)
            loop_image = generated_image
            
        images_per_row = 10
        num_rows = (len(collect_list) + images_per_row - 1) // images_per_row
        single_width = collect_list[0].width
        single_height = collect_list[0].height
        total_width = single_width * min(len(collect_list), images_per_row)
        total_height = single_height * num_rows
        flux_stitched_image = Image.new('RGB', (total_width, total_height))
        for i, img in enumerate(collect_list):
            row = i // images_per_row
            col = i % images_per_row
            flux_stitched_image.paste(img, (col * single_width, row * single_height))
        flux_image = flux_stitched_image
    return flux_image

iface = gr.Interface(
    fn=generate_image,  
    inputs=[gr.Textbox(label="Enter your prompt"), 
        gr.Number(label="Seed", value=2), 
        gr.Slider(0, 10, step=0.1, value=3.5, label="CFG Original"), 
        gr.Slider(0, 10, step=0.1, value=3.5, label="CFG IP"),
        gr.Slider(0, 100, step=1, value=30, label="Num Steps")],
    outputs=gr.Image(label="Generated Image", type="pil")  
)

iface.launch()