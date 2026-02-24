from typing import Optional, Union, List
#from tqdm.notebook import tqdm
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np

#from P2P import ptp_utils
from PIL import Image
import os
import argparse

import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import json
#from utils.real_analysis_pipe import *
from utils.skip_real_pipe import *
import time
from torchvision import transforms
# %%

@torch.no_grad()
def main(
        output_dir='output',
        guidance_scale=7.5,
        K_round=50,
        num_of_ddim_steps=50,
        delta_threshold=5e-12,
        afpi=True,
        fp_th=0.7,
        conv_check=True,
        **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    sample_count = len(os.listdir(output_dir))

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    ldm_stable = MyStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
    inversion = analysis_Inversion(ldm_stable, K_round=K_round, num_ddim_steps=num_of_ddim_steps,
                                                         delta_threshold=delta_threshold, afpi=afpi,
                                                         fp_th=fp_th, conv_check=conv_check)

    with open(f"PIE_bench/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    init_latents, inv_latents, img_latents, rec_latents = [], [], [], []
    cfg_schedules = []
    total_time = 0.0
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    vae = ldm_stable.vae
    for i,(_, item) in enumerate(editing_instruction.items()):
        prompt = item["original_prompt"].replace("[", "").replace("]", "")
        #print(prompt)
        image_path = os.path.join(f"PIE_bench/annotation_images",item["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to("cuda")
        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist
            latent = latent.sample() * 0.18215

        with torch.no_grad():
            de_latent = latent / 0.18215
            decoded = vae.decode(de_latent).sample

        decoded = (decoded / 2 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
        decoded_np = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
        decoded_img = Image.fromarray((decoded_np * 255).round().squeeze().astype("uint8"))

        decoded_img.save(f'{output_dir}/{i}gt.png')

        inversion.method='afpi'
        guidance_scale = 7
        #result_dict = {}
        start_time = time.time()
        inter_inv_latents, fpi_conv_list, cfg_div_list, cfg_schedule = inversion.invert(latent, prompt, guidance_scale=guidance_scale)
        end_time = time.time()
        total_time += (end_time - start_time)
        inv_latent = inter_inv_latents[-1]
        #print(cfg_schedule)
        image_rec, rec_latents_list, rec_div_list = ldm_stable(prompt=prompt, latents=inv_latent, guidance_scale=guidance_scale, cfg_schedule=cfg_schedule)
        image_rec[0].save(f'{output_dir}/{i}rec.png')
        #init_latents.append(init_latent)
        inv_latents.append(inv_latent)
        img_latents.append(latent)
        rec_latents.append(rec_latents_list[-1])
        cfg_schedules.append(cfg_schedule)
        #result_dict['inv_latents'] = inv_latents
        #result_dict['inv_conv_list'] = fpi_conv_list
        #result_dict['inv_div_list'] = cfg_div_list
        #result_dict['rec_latents'] = rec_latents_list
        #result_dict['rec_div_list'] = rec_div_list
        #torch.save(result_dict, f'skip_inv_test/result_dict.pt')
        #torch.save(result_dict, f'{i}/skip_inv_result{guidance_scale}.pt')
    print('total_time:', total_time)
    print('avg_time:', total_time/700)
    #torch.save(init_latents, f'{output_dir}/init_latents.pt')
    torch.save(inv_latents, f'{output_dir}/inv_latents.pt')
    torch.save(img_latents, f'{output_dir}/img_latents.pt')
    torch.save(rec_latents, f'{output_dir}/rec_latents.pt')
    torch.save(cfg_schedules, f'{output_dir}/cfg_schedules.pt')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--K_round",
        type=int,
        default=500,
        help="Optimization Round",
    )
    parser.add_argument(
        "--num_of_ddim_steps",
        type=int,
        default=50,
        help="Blended word needed for P2P",
    )
    parser.add_argument(
        "--delta_threshold",
        type=float,
        default=5e-12,
        help="Delta threshold",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Save editing results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--afpi",
        action='store_true',
    )
    parser.add_argument(
        "--conv_check",
        action='store_true',
    )
    parser.add_argument(
        "--fp_th",
        type=float,
        default=0.7,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = {}
    params['guidance_scale'] = args.guidance_scale
    params['K_round'] = args.K_round
    params['num_of_ddim_steps'] = args.num_of_ddim_steps
    params['delta_threshold'] = args.delta_threshold
    params['conv_check'] = args.conv_check
    params['output_dir'] = args.output
    params['afpi'] = args.afpi
    params['fp_th'] = args.fp_th
    torch.manual_seed(args.seed)
    main(**params)
