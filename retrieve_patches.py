"""
retrieve_patches.py
-------------------
Classify every WSI patch by image-text similarity softmax, then draw a
thumbnail-level region map with one color per label.

Example:
    python retrieve_patches.py \
        --text_feature_dir ./text_features_13_region_specific \
        --h5_dir ./feature_conch \
        --output_dir ./region_maps \
        --slide_folder ./images \
        --file_type svs \
        --level 0 \
        --read_size 448 \
        --logit_scale 100

Inputs:
    text_feature_dir/class_features.npy   shape: (C, 512)
    text_feature_dir/classes.txt          C class names
    text_feature_dir/label_colors.json    optional label colors
    h5_dir/*.h5                           patch features and level-0 coordinates

Outputs per slide:
    <slide>_region_labels.csv
    <slide>_summary.png
    <slide>_class_distribution.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from czi_reader import open_slide


DEFAULT_COLORS = {
    "malignant tissue": (217, 30, 30),
    "benign tissue": (60, 133, 194),
    "stroma": (250, 194, 99),
    "lymphocytes": (90, 186, 125),
    "necrosis": (128, 0, 128),
    "adipose tissue": (245, 237, 203),
    "tissue artifact": (128, 128, 128),
    "blood vessel": (255, 182, 193),
    "extracellular mucin": (0, 255, 255),
    "nerve": (75, 0, 130),
    "hemorrhage": (165, 42, 42),
    "smooth muscle": (210, 105, 30),
    "plasma cells": (255, 20, 147),
}


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-12)


def load_text_features(text_feature_dir: str):
    text_dir = Path(text_feature_dir)
    class_features_path = text_dir / "class_features.npy"
    classes_path = text_dir / "classes.txt"
    colors_path = text_dir / "label_colors.json"

    if not class_features_path.exists():
        raise FileNotFoundError(f"{class_features_path} not found.")
    if not classes_path.exists():
        raise FileNotFoundError(f"{classes_path} not found.")

    class_features = normalize_rows(np.load(class_features_path))
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    if len(classes) != class_features.shape[0]:
        raise ValueError(
            f"classes.txt has {len(classes)} classes, but class_features.npy has "
            f"{class_features.shape[0]} rows."
        )

    colors = dict(DEFAULT_COLORS)
    if colors_path.exists():
        loaded = json.loads(colors_path.read_text())
        colors.update({label: tuple(rgb) for label, rgb in loaded.items()})

    print(f"Loaded text features: {class_features.shape}")
    print(f"Classes: {classes}")
    return class_features, classes, colors


def load_slide_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as handle:
        features = normalize_rows(handle["features"][:])
        coords = handle["coordinates"][:].astype(np.int64)
        patch_indices = handle["patch_indices"][:].astype(np.int64)
        attrs = dict(handle.attrs)
    return features, coords, patch_indices, attrs


def classify_softmax(features: np.ndarray, class_features: np.ndarray, logit_scale: float):
    cosine = features @ class_features.T
    logits = cosine * logit_scale
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-12)
    pred_idx = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(pred_idx)), pred_idx]
    return cosine, probs, pred_idx, confidence


def find_slide_path(slide_folder: str, slide_name: str, file_type: str | None):
    folder = Path(slide_folder)
    candidates = []
    if file_type:
        candidates.append(folder / f"{slide_name}.{file_type.lstrip('.')}")
    for suffix in (".svs", ".ndpi", ".tif", ".tiff", ".czi"):
        candidates.append(folder / f"{slide_name}{suffix}")

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Slide not found for {slide_name} in {slide_folder}")


def make_thumbnail(slide, max_side: int) -> Image.Image:
    full_w, full_h = slide.dimensions
    scale = min(max_side / full_w, max_side / full_h, 1.0)
    size = (max(1, int(full_w * scale)), max(1, int(full_h * scale)))
    return slide.get_thumbnail(size).convert("RGB")


def make_legend(classes: list[str], label_to_color: dict[str, tuple[int, int, int]]) -> Image.Image:
    row_h = 42
    swatch = 28
    width = 520
    height = row_h * len(classes) + 24
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(22)

    y = 12
    for label in classes:
        color = label_to_color.get(label, (0, 0, 0))
        draw.rectangle([14, y + 5, 14 + swatch, y + 5 + swatch], fill=color, outline=(30, 30, 30))
        draw.text((58, y + 5), label, fill=(20, 20, 20), font=font)
        y += row_h
    return image


def draw_legend(classes: list[str], label_to_color: dict[str, tuple[int, int, int]], output_path: Path):
    image = make_legend(classes, label_to_color)
    image.save(output_path)


def save_summary_image(thumbnail: Image.Image, overlay: Image.Image, legend: Image.Image, output_path: Path):
    title_h = 58
    gap = 30
    margin = 24
    panel_h = max(thumbnail.height, overlay.height, legend.height)
    width = thumbnail.width + overlay.width + legend.width + gap * 2 + margin * 2
    height = panel_h + title_h + margin * 2

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(28, bold=True)

    x_thumb = margin
    x_overlay = x_thumb + thumbnail.width + gap
    x_legend = x_overlay + overlay.width + gap
    y_panel = margin + title_h

    draw.text((x_thumb, margin + 4), "(a) Original WSI", fill=(20, 20, 20), font=font)
    draw.text((x_overlay, margin + 4), "(b) Softmax Overlay", fill=(20, 20, 20), font=font)
    draw.text((x_legend, margin + 4), "Labels", fill=(20, 20, 20), font=font)

    canvas.paste(thumbnail, (x_thumb, y_panel))
    canvas.paste(overlay, (x_overlay, y_panel))
    canvas.paste(legend, (x_legend, y_panel))
    canvas.save(output_path)


def draw_region_images(
    slide_path: Path,
    coords: np.ndarray,
    pred_idx: np.ndarray,
    confidence: np.ndarray,
    classes: list[str],
    label_to_color: dict[str, tuple[int, int, int]],
    output_dir: Path,
    slide_name: str,
    level: int,
    read_size: int,
    max_side: int,
    alpha: int,
    min_confidence: float,
):
    slide = open_slide(slide_path)
    try:
        full_w, full_h = slide.dimensions
        downsample = float(slide.level_downsamples[level])
        footprint = read_size * downsample

        thumbnail = make_thumbnail(slide, max_side)
        thumb_w, thumb_h = thumbnail.size
        scale_x = thumb_w / full_w
        scale_y = thumb_h / full_h

        region_map = Image.new("RGBA", thumbnail.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(region_map)
        for (x, y), class_i, conf in zip(coords, pred_idx, confidence):
            if conf < min_confidence:
                continue
            label = classes[int(class_i)]
            color = label_to_color.get(label, (0, 0, 0))
            x0 = int(round(x * scale_x))
            y0 = int(round(y * scale_y))
            x1 = int(round((x + footprint) * scale_x))
            y1 = int(round((y + footprint) * scale_y))
            if x1 <= x0:
                x1 = x0 + 1
            if y1 <= y0:
                y1 = y0 + 1
            draw.rectangle([x0, y0, x1, y1], fill=(*color, alpha))

        overlay = Image.alpha_composite(thumbnail.convert("RGBA"), region_map).convert("RGB")
        legend = make_legend(classes, label_to_color)
        save_summary_image(thumbnail, overlay, legend, output_dir / f"{slide_name}_summary.png")
    finally:
        slide.close()


def write_csv(
    csv_path: Path,
    slide_name: str,
    coords: np.ndarray,
    patch_indices: np.ndarray,
    classes: list[str],
    cosine: np.ndarray,
    probs: np.ndarray,
    pred_idx: np.ndarray,
    confidence: np.ndarray,
):
    fieldnames = [
        "slide_name",
        "patch_idx",
        "x",
        "y",
        "label",
        "softmax_score",
        *[f"cosine_{label}" for label in classes],
        *[f"prob_{label}" for label in classes],
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for coord, patch_idx, cos_row, prob_row, class_i, conf in zip(
            coords, patch_indices, cosine, probs, pred_idx, confidence
        ):
            row = {
                "slide_name": slide_name,
                "patch_idx": int(patch_idx),
                "x": int(coord[0]),
                "y": int(coord[1]),
                "label": classes[int(class_i)],
                "softmax_score": f"{float(conf):.6f}",
            }
            for label, value in zip(classes, cos_row):
                row[f"cosine_{label}"] = f"{float(value):.6f}"
            for label, value in zip(classes, prob_row):
                row[f"prob_{label}"] = f"{float(value):.6f}"
            writer.writerow(row)


def write_distribution(output_path: Path, classes: list[str], pred_idx: np.ndarray):
    counts = np.bincount(pred_idx, minlength=len(classes))
    total = int(counts.sum())
    with open(output_path, "w") as handle:
        handle.write(f"total_patches\t{total}\n")
        for label, count in zip(classes, counts):
            fraction = float(count / total) if total else 0.0
            handle.write(f"{label}\t{int(count)}\t{fraction:.6f}\n")
    return counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify WSI patches into labels using softmax over text similarities."
    )
    parser.add_argument("--text_feature_dir", required=True, help="Directory from encode_text_queries.py")
    parser.add_argument("--h5_dir", required=True, help="Directory containing slide .h5 patch features")
    parser.add_argument("--output_dir", required=True, help="Directory to save label maps")
    parser.add_argument("--slide_folder", required=True, help="Folder containing original WSI files")
    parser.add_argument("--file_type", default=None, help="Slide extension, e.g. svs or ndpi")
    parser.add_argument("--level", type=int, default=0, help="WSI level used during feature extraction")
    parser.add_argument("--read_size", type=int, default=448, help="Read size used during feature extraction")
    parser.add_argument("--logit_scale", type=float, default=100.0, help="Softmax logit scale")
    parser.add_argument("--max_side", type=int, default=1600, help="Max side length for output thumbnail")
    parser.add_argument("--alpha", type=int, default=150, help="Overlay opacity, 0-255")
    parser.add_argument("--min_confidence", type=float, default=0.0, help="Hide patches below this softmax score")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_features, classes, label_to_color = load_text_features(args.text_feature_dir)
    h5_files = sorted(Path(args.h5_dir).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {args.h5_dir}")

    for h5_path in h5_files:
        slide_name = h5_path.stem
        print(f"\nProcessing slide: {slide_name}")
        features, coords, patch_indices, attrs = load_slide_h5(h5_path)
        level = int(attrs.get("extraction_level", args.level))
        read_size = int(attrs.get("read_size", args.read_size))
        slide_path = find_slide_path(args.slide_folder, slide_name, args.file_type)
        if not attrs:
            print("  h5 attrs: none, using --level and --read_size from command line")
        slide = open_slide(slide_path, args.file_type)
        downsample = float(slide.level_downsamples[level])
        slide.close()
        print(
            f"  drawing footprint: level={level}, read_size={read_size}, "
            f"downsample={downsample:.4g}, level0_size={read_size * downsample:.1f}"
        )

        cosine, probs, pred_idx, confidence = classify_softmax(
            features,
            class_features,
            args.logit_scale,
        )
        counts = write_distribution(output_dir / f"{slide_name}_class_distribution.txt", classes, pred_idx)
        write_csv(
            output_dir / f"{slide_name}_region_labels.csv",
            slide_name,
            coords,
            patch_indices,
            classes,
            cosine,
            probs,
            pred_idx,
            confidence,
        )
        draw_region_images(
            slide_path,
            coords,
            pred_idx,
            confidence,
            classes,
            label_to_color,
            output_dir,
            slide_name,
            level,
            read_size,
            args.max_side,
            args.alpha,
            args.min_confidence,
        )

        print(f"  patches: {len(features)}")
        for label, count in zip(classes, counts):
            print(f"  {label:<22} {int(count)}")
        print(f"  CSV:     {output_dir / f'{slide_name}_region_labels.csv'}")
        print(f"  Summary: {output_dir / f'{slide_name}_summary.png'}")

    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
