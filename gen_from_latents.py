"""Generate (reconstruct) images from stored inv_latents using a specified cfg.

Each experiment folder stores inv_latents.pt (the inverted latents).
This script runs the forward diffusion from those latents with a given
--cfg to produce reconstructed images, saved as {i}rec_cfg{X}.png.
Existing files are skipped to avoid duplicate computation.

If cfg_schedules.pt exists in the folder (e.g. from AFPI inversion), the
per-sample schedule is used instead of a fixed cfg — unless --no_schedule
is set.

Usage
-----
  # Fixed cfg
  python gen_from_latents.py --root outputs/migrated/synthetic/cfg7/ddim --cfg 3

  # Use per-sample cfg_schedule (auto-detected)
  python gen_from_latents.py --root outputs/skip_inv --cfg 7

  # Custom output suffix
  python gen_from_latents.py --root ... --cfg 3 --suffix rec_custom.png
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from diffusers import DDIMScheduler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main(args):
    # --- Decide which pipeline to use ---
    cfg_schedule_path = os.path.join(args.root, "cfg_schedules.pt")
    use_cfg_schedule = os.path.exists(cfg_schedule_path) and not args.no_schedule

    if use_cfg_schedule:
        from utils.skip_pipe import MyStableDiffusionPipeline
    else:
        from utils.custom_sd import MyStableDiffusionPipeline

    # --- Load pipeline ---
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    ldm_stable = MyStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", scheduler=scheduler
    ).to(device)

    # --- Load prompts ---
    with open("PIE_bench/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
    prompts = [
        item["original_prompt"].replace("[", "").replace("]", "")
        for _, item in editing_instruction.items()
    ]

    # --- Load inv_latents ---
    inv_latents_path = os.path.join(args.root, "inv_latents.pt")
    if not os.path.exists(inv_latents_path):
        raise FileNotFoundError(f"inv_latents.pt not found in {args.root}")
    inv_latents = torch.load(inv_latents_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(inv_latents)} inv_latents from {inv_latents_path}")

    # --- Load cfg_schedules (optional) ---
    cfg_schedules = None
    if use_cfg_schedule:
        cfg_schedules = torch.load(cfg_schedule_path, map_location="cpu", weights_only=False)
        print(f"Using per-sample cfg_schedules ({len(cfg_schedules)} entries)")
    else:
        print(f"Using fixed guidance_scale = {args.cfg}")

    # --- Generate ---
    suffix = args.suffix
    generated, skipped = 0, 0

    for i in tqdm(range(len(inv_latents)), desc="Generating"):
        out_path = os.path.join(args.root, f"{i}{suffix}")
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        inv_latent = inv_latents[i].to(device)

        if cfg_schedules is not None:
            schedule = list(cfg_schedules[i])          # copy — skip_pipe reverses in-place
            image, _, _ = ldm_stable(
                prompt=prompts[i],
                latents=inv_latent,
                guidance_scale=args.cfg,               # fallback; overridden per step
                cfg_schedule=schedule,
            )
        else:
            image, _ = ldm_stable(
                prompt=prompts[i],
                latents=inv_latent,
                guidance_scale=args.cfg,
            )

        image[0].save(out_path)
        generated += 1

    print(f"\nDone. Generated: {generated}, Skipped (existing): {skipped}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images from stored init_latents."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Experiment folder containing inv_latents.pt",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.0,
        help="Fixed guidance scale for generation (default: 7)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="File suffix for output images (default: rec_cfg{X}.png)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing gen images instead of skipping",
    )
    parser.add_argument(
        "--no_schedule",
        action="store_true",
        help="Ignore cfg_schedules.pt even if present; use fixed --cfg instead",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Default suffix encodes the cfg value: rec_cfg{X}.png
    if args.suffix is None:
        cfg_tag = int(args.cfg) if args.cfg == int(args.cfg) else args.cfg
        args.suffix = f"rec_cfg{cfg_tag}.png"
    torch.manual_seed(args.seed)
    main(args)
