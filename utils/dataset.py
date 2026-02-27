"""PIE-Bench Dataset: synthetic data generation and unified data loading.

Generate synthetic data (run once from project root):
    conda run -n afpi python utils/dataset.py --generate
    # -> saved to PIE_bench/synthetic_data/images.pt

Usage:
    from utils.dataset import get_dataloader

    # Synthetic data
    loader = get_dataloader(mode='synth', batch_size=4)
    for images, prompts, meta in loader:
        # images:  (B, 3, 512, 512) float32 in [0, 1]
        # prompts: list of B strings
        # meta:    dict of lists
        ...

    # Real data
    loader = get_dataloader(mode='real', batch_size=4)
    for images, prompts, meta in loader:
        ...
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file -> project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_MAPPING_FILE = _PROJECT_ROOT / "PIE_bench" / "mapping_file.json"
_ANNOTATION_DIR = _PROJECT_ROOT / "PIE_bench" / "annotation_images"
_SYNTH_DIR = _PROJECT_ROOT / "PIE_bench" / "synthetic_data"
_SYNTH_IMAGES_FILE = _SYNTH_DIR / "images.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_mapping() -> List[Dict[str, Any]]:
    """Load PIE-Bench mapping file, preserving original insertion order."""
    with open(_MAPPING_FILE, "r") as f:
        raw = json.load(f)
    entries = []
    for key, val in raw.items():
        entry = dict(val)
        entry["_key"] = key
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PIEBenchDataset(Dataset):
    """Unified PIE-Bench dataset for synthetic and real images.

    Parameters
    ----------
    mode : str
        ``'synth'`` -- pre-generated images (requires prior generation step).
        ``'real'``  -- original annotation images from PIE-Bench.
    include_mask : bool
        If ``True``, include the RLE-encoded mask in metadata (default False).

    Returns (per sample)
    ----------
    image : Tensor  (3, 512, 512)  float32 in [0, 1]
    prompt : str
    metadata : dict
    """

    def __init__(self, mode: str = "synth", include_mask: bool = False):
        if mode not in ("synth", "real"):
            raise ValueError(f"mode must be 'synth' or 'real', got {mode!r}")
        self.mode = mode
        self.include_mask = include_mask
        self.entries = _load_mapping()
        self._to_tensor = transforms.ToTensor()

        if mode == "synth":
            if not _SYNTH_IMAGES_FILE.exists():
                raise FileNotFoundError(
                    f"Synthetic images not found at {_SYNTH_IMAGES_FILE}.\n"
                    "Run:  python utils/dataset.py --generate"
                )
            self._images = torch.load(
                str(_SYNTH_IMAGES_FILE), map_location="cpu"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        prompt = entry["original_prompt"].replace("[", "").replace("]", "")

        if self.mode == "synth":
            image = self._images[idx]                       # (3, 512, 512) float32
        else:  # real
            path = _ANNOTATION_DIR / entry["image_path"]
            image = self._to_tensor(Image.open(path).convert("RGB"))

        meta: Dict[str, Any] = {
            "index": idx,
            "key": entry["_key"],
            "image_path": entry["image_path"],
            "editing_prompt": entry["editing_prompt"],
            "editing_instruction": entry["editing_instruction"],
            "editing_type_id": entry["editing_type_id"],
            "blended_word": entry["blended_word"],
        }
        if self.include_mask:
            meta["mask"] = entry["mask"]

        return image, prompt, meta


# ---------------------------------------------------------------------------
# Collation & DataLoader
# ---------------------------------------------------------------------------
def _collate_fn(batch):
    """Stack images, keep prompts as list[str], merge metadata dicts."""
    images, prompts, metas = zip(*batch)
    images = torch.stack(images, dim=0)
    prompts = list(prompts)
    meta_batch = {k: [m[k] for m in metas] for k in metas[0]}
    return images, prompts, meta_batch


def get_dataloader(
    mode: str = "synth",
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Create a PIE-Bench DataLoader.

    Note: ``batch_size`` here is for the DataLoader only.  Synthetic image
    *generation* is always sequential (one image at a time) because batched
    UNet / text-encoder forward passes produce different floating-point
    results, breaking determinism with skip_inv.py.

    Parameters
    ----------
    mode : 'synth' or 'real'
    batch_size : int  (default 1)
    shuffle : bool    (default False)
    num_workers : int (default 0)
    **kwargs : forwarded to ``PIEBenchDataset`` (e.g. ``include_mask=True``)
    """
    ds = PIEBenchDataset(mode=mode, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
_SYNTH_PREVIEW_DIR = _SYNTH_DIR / "preview"
_NUM_PREVIEW = 20


@torch.no_grad()
def generate_synthetic_data(shard_size: int = 50):
    """Generate synthetic images for all 700 PIE-Bench entries.

    Uses the same MyStableDiffusionPipeline as skip_inv.py to ensure
    bit-exact determinism.  Images are generated one-at-a-time (matching
    the original sequential loop) and flushed to disk in shards to limit
    RAM usage.

    Reproduces the generation step in skip_inv.py:
      - Pipeline : MyStableDiffusionPipeline  (utils/skip_pipe.py)
      - Model    : CompVis/stable-diffusion-v1-4  +  DDIM  (50 steps)
      - Seed     : 0  (CPU)
      - CFG      : 7
      - Noise    : torch.randn(1, 4, 64, 64) per sample, sequential on CPU
    Images are stored as float32 tensors (no lossy compression).
    The first 20 images are also saved as PNG under synthetic_data/preview/.

    Parameters
    ----------
    shard_size : int
        Number of images per shard file written to disk (default 50).
    """
    from diffusers import DDIMScheduler
    from utils.skip_pipe import MyStableDiffusionPipeline as _Pipeline
    from tqdm import tqdm

    _GEN_GUIDE_SCALE = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determinism (match skip_inv.py)
    torch.manual_seed(0)
    torch.backends.cudnn.allow_tf32 = False

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    pipe = _Pipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        scheduler=scheduler,
    ).to(device)

    entries = _load_mapping()

    _SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    _SYNTH_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    _shard_dir = _SYNTH_DIR / "shards"
    _shard_dir.mkdir(parents=True, exist_ok=True)

    shard_imgs: List[torch.Tensor] = []
    shard_idx = 0

    for idx, entry in enumerate(tqdm(entries, desc="Generating synthetic images")):
        prompt = entry["original_prompt"].replace("[", "").replace("]", "")
        # Generate noise on CPU (same RNG stream as skip_inv.py), then move
        init_latent = torch.randn(1, 4, 64, 64).to(device)

        image_pil, inter_latents, _ = pipe(
            prompt=prompt,
            latents=init_latent,
            guidance_scale=_GEN_GUIDE_SCALE,
        )
        # Convert PIL -> float32 tensor [0, 1], shape (3, 512, 512)
        img_tensor = transforms.ToTensor()(image_pil[0])
        shard_imgs.append(img_tensor)

        # Save preview PNG for first N images
        if idx < _NUM_PREVIEW:
            image_pil[0].save(_SYNTH_PREVIEW_DIR / f"{idx:03d}.png")

        del inter_latents, init_latent
        torch.cuda.empty_cache()

        # Flush shard to disk
        if len(shard_imgs) >= shard_size or idx == len(entries) - 1:
            shard_tensor = torch.stack(shard_imgs, dim=0)
            torch.save(shard_tensor, _shard_dir / f"shard_{shard_idx:04d}.pt")
            shard_imgs.clear()
            shard_idx += 1

    # Unload model to free VRAM/RAM before merge
    del pipe
    torch.cuda.empty_cache()

    # Merge shards into a single file
    print("Merging shards into final images.pt ...")
    shard_files = sorted(_shard_dir.glob("shard_*.pt"))
    all_images: List[torch.Tensor] = []
    for sf in shard_files:
        shard = torch.load(sf, map_location="cpu")
        all_images.extend(shard[j] for j in range(shard.shape[0]))
        del shard

    torch.save(all_images, str(_SYNTH_IMAGES_FILE))
    print(f"Saved {len(all_images)} images -> {_SYNTH_IMAGES_FILE}")

    # Clean up shards
    for sf in shard_files:
        sf.unlink()
    _shard_dir.rmdir()
    print(f"Saved {min(_NUM_PREVIEW, len(all_images))} preview PNGs -> {_SYNTH_PREVIEW_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIE-Bench dataset utilities")
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate synthetic data (seed=0, CompVis/stable-diffusion-v1-4)",
    )
    args = parser.parse_args()

    if args.generate:
        generate_synthetic_data()
    else:
        parser.print_help()
