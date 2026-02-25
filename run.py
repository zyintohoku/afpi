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
from utils.inv_methods import Inversion
import json
from utils.custom_sd import *
import time

# %%

@torch.no_grad()
def main(
        output_dir='output',
        guidance_scale=7.5,
        K_round=50,
        num_of_ddim_steps=50,
        delta_threshold=5e-12,
        #afpi=True,
        method='afpi',
        fp_th=0.7,
        conv_check=True,
        **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    sample_count = len(os.listdir(output_dir))

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    ldm_stable = MyStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
    inversion = Inversion(ldm_stable, K_round=K_round, num_ddim_steps=num_of_ddim_steps, delta_threshold=delta_threshold, method=method, fp_th=fp_th, conv_check=conv_check)

    with open(f"mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    init_latents, inv_latents, gen_latents, rec_latents = [], [], [], []
    total_time = 0.0
    for i,(_, item) in enumerate(editing_instruction.items()):
        prompt = item["original_prompt"].replace("[", "").replace("]", "")
        init_latent = torch.randn(1, 4, 64, 64).to('cuda')
        image_gen, gen_latent = ldm_stable(prompt=prompt, latents=init_latent, guidance_scale=7)
        image_gen[0].save(f'{output_dir}/{i}gen.png')
        start_time = time.time()
        inv_latent = inversion.invert(gen_latent, prompt, guidance_scale)[-1]
        end_time = time.time()
        total_time += (end_time - start_time)
        image_rec, rec_latent = ldm_stable(prompt=prompt, latents=inv_latent, guidance_scale=guidance_scale)
        image_rec[0].save(f'{output_dir}/{i}rec.png')
        #print(F.mse_loss(init_latent, inv_latent).item())
        init_latents.append(init_latent)
        inv_latents.append(inv_latent)
        gen_latents.append(gen_latent)
        rec_latents.append(rec_latent)
    print('total_time:', total_time)
    print('avg_time:', total_time/700)
    torch.save(init_latents, f'{output_dir}/init_latents.pt')
    torch.save(inv_latents, f'{output_dir}/inv_latents.pt')
    torch.save(gen_latents, f'{output_dir}/gen_latents.pt')
    torch.save(rec_latents, f'{output_dir}/rec_latents.pt')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--K_round",
        type=int,
        default=50,
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
        default=7.5,
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
        "--method",
        type=str,
        default="afpi",
        help="Inversion method: afpi, exact, spd, fpi, aidi, ddim, newton",
    )
    #parser.add_argument(
    #    "--afpi",
    #    action='store_true',
    #)
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
    #params['afpi'] = args.afpi
    params['method'] = args.method
    params['fp_th'] = args.fp_th
    torch.manual_seed(args.seed)
    main(**params)
