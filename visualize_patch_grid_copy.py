"""
visualize_patch_grid.py
-----------------------
在 WSI 缩略图上叠加 patch 网格，扫描逻辑与提特征代码完全一致。

核心：在中间分辨率（level 5）空间扫描 mask，避免直接用 level 1（5GB）
或缩略图（精度太低），然后将格子坐标换算回缩略图空间画框。

用法：
  # 快速模式（只看 mask）
  python visualize_patch_grid.py \
      --slide slide.tif --mask mask.tif --output out.png \
      --read_size 256 --level 1

  # 精确模式（含 blank 过滤，和提特征完全一致）
  python visualize_patch_grid.py \
      --slide slide.tif --mask mask.tif --output out.png \
      --read_size 256 --level 1 --blank_threshold 0.6

  # 随机保存 patch（默认 100 张，可用 --save_patches 指定数量）
  python visualize_patch_grid.py \
      --slide slide.tif --mask mask.tif --output out.png \
      --read_size 256 --level 1 --save_patches 100 --patch_dir ./patches
"""

import os
import random
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import openslide


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--slide',           required=True)
    p.add_argument('--mask',            required=True)
    p.add_argument('--output',          default='grid_test.png')
    p.add_argument('--read_size',       type=int, default=None)
    p.add_argument('--patch_size',      type=int, default=224)
    p.add_argument('--level',           type=int, default=0,
                   help='提特征时用的 WSI level（默认 0）')
    p.add_argument('--thumb_size',      type=int, default=1000)
    p.add_argument('--blank_threshold', type=float, default=None)
    # ── 新增参数 ──────────────────────────────────────────────
    p.add_argument('--save_patches',    type=int,  default=None,
                   help='随机保存 N 张 patch（默认不保存）')
    p.add_argument('--patch_dir',       type=str,  default='saved_patches',
                   help='patch 保存目录（默认 saved_patches/）')
    p.add_argument('--seed',            type=int,  default=42,
                   help='随机种子，保证可复现（默认 42）')
    return p.parse_args()


def is_blank(patch_np, threshold):
    gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
    return np.sum(gray > 230) / gray.size > threshold


def save_random_patches(slide, candidates_scan, scan_ds, feat_level,
                        read_size, patch_size, n, patch_dir, seed):
    """
    从 scan_level 坐标列表里随机抽 n 个，读取对应 patch 并保存为 PNG。

    Parameters
    ----------
    slide          : openslide.OpenSlide
    candidates_scan: list of (sx, sy)  ← scan_level 坐标
    scan_ds        : slide.level_downsamples[scan_level]
    feat_level     : 提特征用的 level（read_region 的 level 参数）
    read_size      : 在 feat_level 读取的像素尺寸
    patch_size     : 最终保存的正方形大小
    n              : 要保存的 patch 数量
    patch_dir      : 输出目录
    seed           : 随机种子
    """
    os.makedirs(patch_dir, exist_ok=True)

    # 如果候选数不足 n，则全部保存
    n_actual = min(n, len(candidates_scan))
    if n_actual < n:
        print(f"  [警告] 候选 patch 仅 {len(candidates_scan)} 个，"
              f"将全部保存（请求 {n} 张）")

    random.seed(seed)
    sampled = random.sample(candidates_scan, n_actual)

    print(f"\n  Saving {n_actual} random patches -> {patch_dir}/")
    for i, (sx, sy) in enumerate(tqdm(sampled, unit='patch')):
        # scan_level 坐标换算回 level-0（read_region 要求）
        x0 = int(round(sx * scan_ds))
        y0 = int(round(sy * scan_ds))
        try:
            region = slide.read_region(
                (x0, y0), feat_level, (read_size, read_size)
            ).convert('RGB')
            if patch_size != read_size:
                region = region.resize(
                    (patch_size, patch_size), Image.BILINEAR
                )
            fname = os.path.join(patch_dir, f"patch_{i:04d}_x{x0}_y{y0}.png")
            region.save(fname)
        except Exception as e:
            print(f"  [跳过] patch {i} ({x0},{y0}) 读取失败: {e}")

    print(f"  Done. {n_actual} patches saved to '{patch_dir}/'")


def main():
    args = parse_args()

    # ── 打开 WSI ──────────────────────────────────────────────
    print(f"Opening slide : {os.path.basename(args.slide)}")
    slide      = openslide.OpenSlide(args.slide)
    num_levels = slide.level_count
    feat_level = min(args.level, num_levels - 1)
    feat_dim   = slide.level_dimensions[feat_level]  # (W, H)
    feat_ds    = slide.level_downsamples[feat_level]

    mpp_x  = slide.properties.get('openslide.mpp-x')
    mpp_y  = slide.properties.get('openslide.mpp-y')
    avg_mpp = (float(mpp_x) + float(mpp_y)) / 2 if (mpp_x and mpp_y) else None
    mag     = ("40x" if avg_mpp < 0.3 else "20x") if avg_mpp else "Unknown"

    read_size = args.read_size or (512 if mag == "40x" else 256)
    step      = read_size

    print(f"  mag={mag}  feat_level={feat_level}  "
          f"feat_dim={feat_dim}  read_size={read_size}")

    # ── 选合适的扫描 level ────────────────────────────────────
    scan_level = feat_level
    for lv in range(feat_level, num_levels):
        lv_ds   = slide.level_downsamples[lv]
        lv_dim  = slide.level_dimensions[lv]
        ratio   = lv_ds / feat_ds
        step_lv = step / ratio
        mem_mb  = lv_dim[0] * lv_dim[1] / 1e6
        if step_lv >= 4 and mem_mb < 200:
            scan_level = lv
            break

    scan_dim = slide.level_dimensions[scan_level]
    scan_ds  = slide.level_downsamples[scan_level]
    step_scan = max(1, int(round(step * feat_ds / scan_ds)))

    print(f"  scan_level={scan_level}  scan_dim={scan_dim}  "
          f"step_scan={step_scan} px  "
          f"(mem ~{scan_dim[0]*scan_dim[1]/1e6:.0f} MB)")

    # ── 缩略图 ────────────────────────────────────────────────
    scale   = min(args.thumb_size / feat_dim[0], args.thumb_size / feat_dim[1])
    thumb_w = int(feat_dim[0] * scale)
    thumb_h = int(feat_dim[1] * scale)
    print(f"  thumbnail : {thumb_w}x{thumb_h}  scale={scale:.5f}")
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert('RGB')

    # ── 读 PNG mask ───────────────────────────────────────────
    print(f"Loading mask  : {os.path.basename(args.mask)}")
    arr = np.array(Image.open(args.mask).convert('L'))  # 灰度，shape=(H, W)
    mask = cv2.resize((arr > 0).astype(np.uint8),
                      (scan_dim[0], scan_dim[1]),
                      interpolation=cv2.INTER_NEAREST)
    print(f"  mask native {arr.shape[1]}x{arr.shape[0]} "
          f"-> scan space {scan_dim[0]}x{scan_dim[1]}")

    # ── 在 scan_level 空间扫描 ────────────────────────────────
    candidates = []  # (sx, sy) in scan_level coords
    for sy in range(0, scan_dim[1], step_scan):
        for sx in range(0, scan_dim[0], step_scan):
            if np.sum(mask[sy:sy + step_scan, sx:sx + step_scan]) > 0:
                candidates.append((sx, sy))

    total_grid = (scan_dim[1] // step_scan) * (scan_dim[0] // step_scan)
    bg_skipped = total_grid - len(candidates)

    # ── Blank 过滤 ────────────────────────────────────────────
    kept_boxes     = []
    filtered_boxes = []
    kept_coords    = []  # ← 与 kept_boxes 同步记录 scan_level 坐标

    def scan_to_thumb(sx, sy):
        fx = sx * scan_ds / feat_ds
        fy = sy * scan_ds / feat_ds
        tx = int(fx * scale)
        ty = int(fy * scale)
        tw = max(1, int(step * scale))
        return (tx, ty, tx + tw, ty + tw)

    if args.blank_threshold is None:
        for (sx, sy) in candidates:
            kept_boxes.append(scan_to_thumb(sx, sy))
            kept_coords.append((sx, sy))
    else:
        print(f"\n  Applying blank filter (threshold={args.blank_threshold})...")
        for (sx, sy) in tqdm(candidates, unit='patch'):
            x0 = int(round(sx * scan_ds))
            y0 = int(round(sy * scan_ds))
            box = scan_to_thumb(sx, sy)
            try:
                region = slide.read_region(
                    (x0, y0), feat_level, (read_size, read_size)
                ).convert('RGB').resize(
                    (args.patch_size, args.patch_size), Image.BILINEAR
                )
                if is_blank(np.array(region), args.blank_threshold):
                    filtered_boxes.append(box)
                else:
                    kept_boxes.append(box)
                    kept_coords.append((sx, sy))
            except Exception:
                filtered_boxes.append(box)

    # ── 统计 ──────────────────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"  step (feat_level pixels) : {step}")
    print(f"  step (scan_level pixels) : {step_scan}")
    print(f"  Total grid positions     : {total_grid}")
    print(f"  Background (mask=0)      : {bg_skipped}")
    print(f"  Tissue candidates        : {len(candidates)}")
    if args.blank_threshold is not None:
        print(f"  Blank filtered out       : {len(filtered_boxes)}"
              f"  (threshold={args.blank_threshold})")
        print(f"  Final kept patches       : {len(kept_boxes)}")
    print(f"{'='*45}")

    # ── 随机保存 patch（新增） ────────────────────────────────
    if args.save_patches is not None:
        save_random_patches(
            slide       = slide,
            candidates_scan = kept_coords,   # 只从通过过滤的 patch 里采样
            scan_ds     = scan_ds,
            feat_level  = feat_level,
            read_size   = read_size,
            patch_size  = args.patch_size,
            n           = args.save_patches,
            patch_dir   = args.patch_dir,
            seed        = args.seed,
        )

    slide.close()

    # ── 绘制缩略图 ────────────────────────────────────────────
    canvas = thumb.copy()
    draw   = ImageDraw.Draw(canvas, 'RGBA')

    for box in kept_boxes:
        draw.rectangle(list(box), outline=(0, 0, 0), width=1)

    canvas.save(args.output)
    print(f"\nSaved grid -> {args.output}")


if __name__ == '__main__':
    main()