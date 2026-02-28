import os
import re
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

@dataclass
class PairResult:
    stem: str
    ssim: float
    psnr: float
    mse: float


def load_rgb(path: Path, resize_to: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if resize_to is not None:
        img = img.resize(resize_to, Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def compute_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    # a,b: float32 in [0,1], shape (H,W,3)
    mse_val = float(np.mean((a - b) ** 2))
    psnr_val = float(psnr(a, b, data_range=1.0))
    ssim_val = float(ssim(a, b, channel_axis=2, data_range=1.0))
    return ssim_val, psnr_val, mse_val


def build_pairs_diff_dir(
    rec_dir: Path,
    gen_dir: Path,
    suffix_gen: str = "_gen.png",
    suffix_rec: str = "_rec.png",
) -> List[Tuple[str, Path, Path]]:
    gen_dir = Path(gen_dir)
    rec_dir = Path(rec_dir)

    # 1) gen_dir 里只拿 *_gen.png（忽略同目录里的 *_rec.png）
    gen_files: Dict[str, Path] = {}
    for p in gen_dir.glob(f"*{suffix_gen}"):
        stem = p.name[: -len(suffix_gen)]
        gen_files[stem] = p

    # 2) rec_dir 里拿 *_rec.png
    rec_files: Dict[str, Path] = {}
    for p in rec_dir.glob(f"*{suffix_rec}"):
        stem = p.name[: -len(suffix_rec)]
        rec_files[stem] = p

    # 3) 取交集
    common = sorted(set(gen_files.keys()) & set(rec_files.keys()))
    pairs = [(s, gen_files[s], rec_files[s]) for s in common]

    # 打印匹配/缺失统计
    missing_rec = sorted(set(gen_files.keys()) - set(rec_files.keys()))
    extra_rec = sorted(set(rec_files.keys()) - set(gen_files.keys()))

    print(f"gen_dir 中 *_gen.png 数量: {len(gen_files)}")
    print(f"rec_dir 中 *_rec.png 数量: {len(rec_files)}")
    print(f"成功匹配的对数: {len(pairs)}")
    print(f"gen 有但 rec_dir 缺失: {len(missing_rec)}")
    print(f"rec_dir 多出来(无对应 gen): {len(extra_rec)}")

    # 可选：把缺失的前几个打印出来方便排查
    if missing_rec:
        print("缺失 rec 的示例(最多10个):", missing_rec[:10])

    return pairs

def build_pairs_single_dir(
    root: Path,
    suffix_a: str = "gen.png",
    suffix_b: str = "rec.png",
) -> List[Tuple[str, Path, Path]]:
    files = list(root.glob("*.png"))
    map_a: Dict[str, Path] = {}
    map_b: Dict[str, Path] = {}

    for p in files:
        name = p.name
        if name.endswith(suffix_a):
            stem = name[: -len(suffix_a)]
            map_a[stem] = p
        elif name.endswith(suffix_b):
            stem = name[: -len(suffix_b)]
            map_b[stem] = p

    stems = sorted(set(map_a.keys()) & set(map_b.keys()))
    pairs = [(s, map_a[s], map_b[s]) for s in stems]
    return pairs


def summarize_thresholds(values: np.ndarray, thresholds: List[float], mode: str) -> List[Tuple[float, float]]:
    """
    mode:
      - "ge": proportion >= threshold
      - "le": proportion <= threshold
    """
    out = []
    n = len(values)
    for t in thresholds:
        if mode == "ge":
            p = float(np.mean(values >= t)) if n else 0.0
        elif mode == "le":
            p = float(np.mean(values <= t)) if n else 0.0
        else:
            raise ValueError("mode must be 'ge' or 'le'")
        out.append((t, p))
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="包含_png 对的文件夹")
    parser.add_argument("--gen_dir", type=str, default="", help="独立的 gen 图文件夹（为空则在 root 里找 gen）")
    parser.add_argument("--suffix_a", type=str, default="gen.png", help="A 图后缀")
    parser.add_argument("--suffix_b", type=str, default="rec.png", help="B 图后缀")
    parser.add_argument("--resize", type=str, default="", help="可选: WxH，比如 512x512；为空则不 resize")
    parser.add_argument("--out_csv", type=str, default="pair_metrics.csv")
    parser.add_argument("--out_summary", type=str, default="threshold_summary.csv")
    parser.add_argument("--ssim_th", type=float, nargs=2, default=[0.70, 0.98],
                        metavar=("MIN", "MAX"), help="SSIM threshold range [min, max]")
    parser.add_argument("--psnr_th", type=float, nargs=2, default=[20, 40],
                        metavar=("MIN", "MAX"), help="PSNR threshold range [min, max]")
    parser.add_argument("--mse_th", type=float, nargs=2, default=[0.001, 0.010],
                        metavar=("MIN", "MAX"), help="MSE threshold range [min, max]")
    parser.add_argument("--n_steps", type=int, default=10,
                        help="Number of threshold intervals for all metrics")
    args = parser.parse_args()

    root = Path(args.root)
    assert root.exists(), f"not found: {root}"

    # Put outputs into a 'result' subfolder under root
    result_dir = root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_csv = result_dir / args.out_csv
    out_summary = result_dir / args.out_summary

    resize_to = None
    if args.resize.strip():
        m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", args.resize.strip().lower())
        if not m:
            raise ValueError("--resize 需形如 512x512")
        w, h = int(m.group(1)), int(m.group(2))
        resize_to = (w, h)

    if args.gen_dir:
        gen_dir = Path(args.gen_dir)
        assert gen_dir.exists(), f"gen_dir not found: {gen_dir}"
        pairs = build_pairs_diff_dir(root, gen_dir, args.suffix_a, args.suffix_b)
    else:
        pairs = build_pairs_single_dir(root, args.suffix_a, args.suffix_b)
    if len(pairs) == 0:
        raise RuntimeError("没有找到任何配对。请检查文件名后缀/路径。")

    results: List[PairResult] = []
    # 可选：固定 resize 到第一张图大小，避免尺寸不一致
    # 如果你不想 resize，但尺寸可能不同，可以改成强制 resize_to=第一张
    for stem, pa, pb in tqdm(pairs, desc=f"Pairs ({len(pairs)})"):
        a = load_rgb(pa, resize_to=resize_to)
        b = load_rgb(pb, resize_to=resize_to)
        if a.shape != b.shape:
            # 自动对齐到 a 的尺寸
            b = np.asarray(Image.fromarray((b * 255).astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.BICUBIC)).astype(np.float32) / 255.0

        ssim_val, psnr_val, mse_val = compute_metrics(a, b)
        results.append(PairResult(stem=stem, ssim=ssim_val, psnr=psnr_val, mse=mse_val))

    # 导出每对的指标
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "ssim", "psnr", "mse"])
        for r in results:
            w.writerow([r.stem, f"{r.ssim:.6f}", f"{r.psnr:.4f}", f"{r.mse:.8f}"])

    ssim_vals = np.array([r.ssim for r in results], dtype=np.float32)
    psnr_vals = np.array([r.psnr for r in results], dtype=np.float32)
    mse_vals = np.array([r.mse for r in results], dtype=np.float32)

    n = args.n_steps + 1  # n_steps intervals -> n_steps+1 points
    ssim_th = np.linspace(args.ssim_th[0], args.ssim_th[1], n).tolist()
    psnr_th = np.linspace(args.psnr_th[0], args.psnr_th[1], n).tolist()
    mse_th = np.linspace(args.mse_th[0], args.mse_th[1], n).tolist()

    ssim_summary = summarize_thresholds(ssim_vals, ssim_th, mode="ge")
    psnr_summary = summarize_thresholds(psnr_vals, psnr_th, mode="ge")
    mse_summary = summarize_thresholds(mse_vals, mse_th, mode="le")

    # 导出阈值统计
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "threshold", "condition", "ratio", "count", "total"])
        total = len(results)

        for t, p in ssim_summary:
            cnt = int(round(p * total))
            w.writerow(["SSIM", t, ">=t", f"{p:.6f}", cnt, total])

        for t, p in psnr_summary:
            cnt = int(round(p * total))
            w.writerow(["PSNR", t, ">=t", f"{p:.6f}", cnt, total])

        for t, p in mse_summary:
            cnt = int(round(p * total))
            w.writerow(["MSE", t, "<=t", f"{p:.6f}", cnt, total])

    print(f"[OK] pairs: {len(results)}")
    print(f"[OK] per-pair saved to: {out_csv}")
    print(f"[OK] threshold summary saved to: {out_summary}")


if __name__ == "__main__":
    main()

