"""
retrieve_top_class_patches.py
-----------------------------
Export the top-K WSI patches for one text-defined histology class.

Example:
    python retrieve_top_class_patches.py \
        --text_feature_dir text_features \
        --h5_dir feature_conch \
        --slide_folder image \
        --output_dir top_class_patches \
        --label "stroma" \
        --topk 100 \
        --file_type svs \
        --level 1 \
        --read_size 224 \
        --patch_size 224 \
        --logit_scale 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import openslide
from PIL import Image, ImageDraw


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-12)


def load_text_features(text_feature_dir: str):
    text_dir = Path(text_feature_dir)
    class_features = normalize_rows(np.load(text_dir / "class_features.npy"))
    classes = [line.strip() for line in (text_dir / "classes.txt").read_text().splitlines() if line.strip()]
    if len(classes) != class_features.shape[0]:
        raise ValueError("classes.txt and class_features.npy have different class counts.")
    return class_features, classes


def choose_label(classes: list[str], requested_label: str | None) -> str:
    if requested_label:
        if requested_label not in classes:
            raise ValueError(f"Unknown label: {requested_label}. Available labels: {classes}")
        return requested_label

    print("\nAvailable labels:")
    for i, label in enumerate(classes):
        print(f"  [{i}] {label}")
    choice = input("\nWhich label do you want to retrieve top patches for? ").strip()
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(classes):
            return classes[idx]
    if choice in classes:
        return choice
    raise ValueError(f"Invalid label choice: {choice}")


def softmax(scores: np.ndarray, logit_scale: float) -> np.ndarray:
    logits = scores * logit_scale
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-12)


def find_slide_path(slide_folder: str, slide_name: str, file_type: str | None):
    folder = Path(slide_folder)
    candidates = []
    if file_type:
        candidates.append(folder / f"{slide_name}.{file_type.lstrip('.')}")
    for suffix in (".svs", ".ndpi", ".tif", ".tiff"):
        candidates.append(folder / f"{slide_name}{suffix}")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Slide not found for {slide_name} in {slide_folder}")


def safe_name(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")


def load_all_candidates(
    h5_dir: str,
    class_features: np.ndarray,
    classes: list[str],
    label: str,
    logit_scale: float,
    rank_by: str,
):
    class_idx = classes.index(label)
    candidates = []

    for h5_path in sorted(Path(h5_dir).glob("*.h5")):
        slide_name = h5_path.stem
        with h5py.File(h5_path, "r") as handle:
            features = normalize_rows(handle["features"][:])
            coords = handle["coordinates"][:].astype(np.int64)
            patch_indices = handle["patch_indices"][:].astype(np.int64)
            attrs = dict(handle.attrs)

        cosine = features @ class_features.T
        probs = softmax(cosine, logit_scale)
        score_values = probs[:, class_idx] if rank_by == "softmax" else cosine[:, class_idx]

        for row_idx, score in enumerate(score_values):
            candidates.append(
                {
                    "slide_name": slide_name,
                    "patch_idx": int(patch_indices[row_idx]),
                    "x": int(coords[row_idx, 0]),
                    "y": int(coords[row_idx, 1]),
                    "score": float(score),
                    "cosine": float(cosine[row_idx, class_idx]),
                    "softmax": float(probs[row_idx, class_idx]),
                    "level": int(attrs.get("extraction_level", -1)),
                    "read_size": int(attrs.get("read_size", -1)),
                }
            )

        print(f"Loaded {slide_name}: {len(features)} patches")

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def read_patch(slide, x: int, y: int, level: int, read_size: int, patch_size: int) -> Image.Image:
    patch = slide.read_region((x, y), level, (read_size, read_size)).convert("RGB")
    if read_size != patch_size:
        patch = patch.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
    return patch


def save_grid(saved: list[tuple[Image.Image, dict]], output_path: Path, patch_size: int):
    if not saved:
        return
    cols = 10
    label_h = 24
    rows = int(np.ceil(len(saved) / cols))
    grid = Image.new("RGB", (cols * patch_size, rows * (patch_size + label_h)), "white")
    draw = ImageDraw.Draw(grid)
    for i, (patch, item) in enumerate(saved):
        col = i % cols
        row = i // cols
        x0 = col * patch_size
        y0 = row * (patch_size + label_h)
        grid.paste(patch, (x0, y0))
        draw.text((x0 + 4, y0 + patch_size + 4), f"{i + 1:03d} {item['score']:.3f}", fill=(20, 20, 20))
    grid.save(output_path)


def export_top_patches(
    candidates: list[dict],
    slide_folder: str,
    file_type: str | None,
    output_dir: Path,
    label: str,
    topk: int,
    level: int,
    read_size: int,
    patch_size: int,
):
    label_dir = output_dir / f"{safe_name(label)}_top{topk}"
    label_dir.mkdir(parents=True, exist_ok=True)

    slide_cache = {}
    saved = []
    manifest_path = label_dir / "manifest.csv"

    with open(manifest_path, "w", newline="") as handle:
        fieldnames = [
            "rank",
            "patch_file",
            "slide_name",
            "patch_idx",
            "x",
            "y",
            "score",
            "cosine",
            "softmax",
            "level",
            "read_size",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for rank, item in enumerate(candidates[:topk], start=1):
            slide_name = item["slide_name"]
            if slide_name not in slide_cache:
                slide_path = find_slide_path(slide_folder, slide_name, file_type)
                slide_cache[slide_name] = openslide.OpenSlide(str(slide_path))

            patch_level = item["level"] if item["level"] >= 0 else level
            patch_read_size = item["read_size"] if item["read_size"] > 0 else read_size
            patch = read_patch(
                slide_cache[slide_name],
                item["x"],
                item["y"],
                patch_level,
                patch_read_size,
                patch_size,
            )

            patch_file = (
                f"rank{rank:03d}_score{item['score']:.6f}_"
                f"cos{item['cosine']:.6f}_prob{item['softmax']:.6f}_"
                f"{slide_name}_idx{item['patch_idx']}_x{item['x']}_y{item['y']}.png"
            )
            patch.save(label_dir / patch_file)
            saved.append((patch.copy(), item))
            writer.writerow({"rank": rank, "patch_file": patch_file, **item})

    for slide in slide_cache.values():
        slide.close()

    save_grid(saved, label_dir / f"{safe_name(label)}_top{len(saved)}_grid.png", patch_size)
    return label_dir, len(saved), manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Export top-K patches for one class.")
    parser.add_argument("--text_feature_dir", required=True)
    parser.add_argument("--h5_dir", required=True)
    parser.add_argument("--slide_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default=None, help="Class name. If omitted, choose interactively.")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--file_type", default=None)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--read_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--logit_scale", type=float, default=10.0)
    parser.add_argument("--rank_by", choices=["softmax", "cosine"], default="softmax")
    return parser.parse_args()


def main():
    args = parse_args()
    class_features, classes = load_text_features(args.text_feature_dir)
    label = choose_label(classes, args.label)

    print(f"\nRetrieving top-{args.topk} patches for label: {label}")
    print(f"Ranking by: {args.rank_by}")
    candidates = load_all_candidates(
        args.h5_dir,
        class_features,
        classes,
        label,
        args.logit_scale,
        args.rank_by,
    )
    label_dir, saved_count, manifest_path = export_top_patches(
        candidates,
        args.slide_folder,
        args.file_type,
        Path(args.output_dir),
        label,
        args.topk,
        args.level,
        args.read_size,
        args.patch_size,
    )

    print(f"\nSaved {saved_count} patches to: {label_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Grid: {label_dir / f'{safe_name(label)}_top{saved_count}_grid.png'}")


if __name__ == "__main__":
    main()
