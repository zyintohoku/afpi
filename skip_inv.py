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
from utils.skip_pipe import *
import time
from torchvision import transforms
# %%

_GEN_GUIDE_SCALE = 7
torch.backends.cudnn.allow_tf32 = False

@torch.no_grad()
def main(
        output_dir='output',
        guidance_scale=7.5,
        K_round=50,
        num_of_ddim_steps=50,
        delta_threshold=5e-12,
        **kwargs
):
    os.makedirs(output_dir, exist_ok=True)

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    ldm_stable = MyStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
    inversion = analysis_Inversion(ldm_stable, K_round=K_round, num_ddim_steps=num_of_ddim_steps,
                                                         delta_threshold=delta_threshold)

    with open(f"PIE_bench/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    init_latents, inv_latents, gen_latents, rec_latents = [], [], [], []
    cfg_schedules = []
    total_time = 0.0
    for i,(_, item) in enumerate(tqdm(editing_instruction.items())):
        init_latent = torch.randn(1, 4, 64, 64).to('cuda')
        if i<17:
            continue
        prompt = item["original_prompt"].replace("[", "").replace("]", "")
        image_gen, inter_latents, _ = ldm_stable(prompt=prompt, latents=init_latent, guidance_scale=_GEN_GUIDE_SCALE)
        image_gen[0].save(f'{output_dir}/{i}gen.png')
        gen_latent = inter_latents[-1]
        assert inversion.method in ['afpi']
        start_time = time.time()
        inter_inv_latents, _, _, cfg_schedule = inversion.invert(gen_latent, prompt, guidance_scale=_GEN_GUIDE_SCALE)
        end_time = time.time()
        total_time += (end_time - start_time)
        inv_latent = inter_inv_latents[-1]
        image_rec, rec_latents_list, _ = ldm_stable(prompt=prompt, latents=inv_latent, guidance_scale=None, cfg_schedule=cfg_schedule)
        image_rec[0].save(f'{output_dir}/{i}rec.png')
        
        init_latents.append(init_latent)
        inv_latents.append(inv_latent)
        gen_latents.append(gen_latent)
        rec_latents.append(rec_latents_list[-1])
        cfg_schedules.append(cfg_schedule)
    print('total_time:', total_time,'time/sample:', total_time/700)
    torch.save(init_latents, f'{output_dir}/init_latents.pt')
    torch.save(inv_latents, f'{output_dir}/inv_latents.pt')
    torch.save(gen_latents, f'{output_dir}/gen_latents.pt')
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
        default=5e-13,
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = {}
    params['guidance_scale'] = args.guidance_scale
    params['K_round'] = args.K_round
    params['num_of_ddim_steps'] = args.num_of_ddim_steps
    params['delta_threshold'] = args.delta_threshold
    params['output_dir'] = args.output
    torch.manual_seed(args.seed)
    main(**params)
