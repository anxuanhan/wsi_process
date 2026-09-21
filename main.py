#!/usr/bin/env python3
"""
FastAPI WSI region analysis app.

Run:
    python main.py

Then open:
    http://127.0.0.1:8000
"""

from __future__ import annotations

import html
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from czi_reader import open_slide


APP_DIR = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


JOBS_DIR = _env_path("WSI_JOBS_DIR", APP_DIR / "web_jobs")
CACHE_DIR = _env_path("WSI_CACHE_DIR", APP_DIR / "web_cache")
JOBS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHECKPOINT_PATH = _env_path(
    "CONCH_CHECKPOINT_PATH",
    APP_DIR / "checkpoints" / "conch" / "pytorch_model.bin",
)
DEFAULT_REFERENCE_PATCH = _env_path(
    "REFERENCE_PATCH_PATH",
    APP_DIR / "assets" / "reference_patch.png",
)

DEFAULT_PATCH_SIZE = 224
DEFAULT_READ_SIZE = 448
DEFAULT_LEVEL = 0
DEFAULT_MIN_AREA = 1_000_000
DEFAULT_MIN_HOLE = 100_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 2
DEFAULT_BLANK_THRESHOLD = 0.6
DEFAULT_LOGIT_SCALE = 10.0
DEFAULT_MAX_SIDE = 1000

CZI_PATCH_SIZE = 224
CZI_READ_SIZE = 656
CZI_BATCH_SIZE = 128
CZI_NUM_WORKERS = 16
CZI_STRIP_PATCHES = 32
CZI_BLANK_THRESHOLD = 0.3
CZI_MIN_MASK_RATIO = 0.7
CZI_PREVIEW_LEVEL = 6

ALLOWED_EXTS = {".svs", ".ndpi", ".tif", ".tiff", ".czi"}

app = FastAPI(title="WSI Region Analyzer")
app.mount("/jobs", StaticFiles(directory=str(JOBS_DIR)), name="jobs")

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return JOBS[job_id]


def _update_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def _append_log(job_id: str, text: str):
    with JOBS_LOCK:
        JOBS[job_id].setdefault("log", "")
        JOBS[job_id]["log"] += text
        JOBS[job_id]["log"] = JOBS[job_id]["log"][-20000:]


def _job_url(path: Path) -> str:
    return "/jobs/" + path.relative_to(JOBS_DIR).as_posix()


def _safe_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported WSI type: {ext}")
    return ext.lstrip(".")


def _make_thumbnail(slide_path: Path, output_path: Path, max_side: int = 1200):
    slide = open_slide(slide_path)
    try:
        full_w, full_h = slide.dimensions
        scale = min(max_side / full_w, max_side / full_h, 1.0)
        size = (max(1, int(full_w * scale)), max(1, int(full_h * scale)))
        thumb = slide.get_thumbnail(size).convert("RGB")
        thumb.save(output_path)
    finally:
        slide.close()


def _run_step(job_id: str, name: str, cmd: list[str], cwd: Path):
    _append_log(job_id, f"\n\n===== {name} =====\n")
    _append_log(job_id, " ".join(cmd) + "\n\n")
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _append_log(job_id, line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{name} failed with exit code {rc}")


def _require_file(path: Path, env_name: str, description: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}. Set {env_name} to the correct file path."
        )


def _cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _slide_cache_payload(slide_path: Path, file_type: str) -> dict[str, Any]:
    stat = slide_path.stat()
    is_czi = file_type == "czi"
    return {
        "name": slide_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "file_type": file_type,
        "patch_size": CZI_PATCH_SIZE if is_czi else DEFAULT_PATCH_SIZE,
        "read_size": CZI_READ_SIZE if is_czi else DEFAULT_READ_SIZE,
        "level": DEFAULT_LEVEL,
        "min_area": DEFAULT_MIN_AREA,
        "min_hole": DEFAULT_MIN_HOLE,
        "blank_threshold": CZI_BLANK_THRESHOLD if is_czi else DEFAULT_BLANK_THRESHOLD,
        "min_mask_ratio": CZI_MIN_MASK_RATIO if is_czi else 0.0,
        "reference_patch": str(DEFAULT_REFERENCE_PATCH),
    }


def _text_cache_payload() -> dict[str, Any]:
    encode_script = APP_DIR / "encode_text_queries.py"
    checkpoint_stat = DEFAULT_CHECKPOINT_PATH.stat() if DEFAULT_CHECKPOINT_PATH.exists() else None
    script_stat = encode_script.stat()
    return {
        "encode_script_mtime_ns": script_stat.st_mtime_ns,
        "encode_script_size": script_stat.st_size,
        "checkpoint": str(DEFAULT_CHECKPOINT_PATH),
        "checkpoint_size": checkpoint_stat.st_size if checkpoint_stat else None,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns if checkpoint_stat else None,
    }


def _link_or_copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_tree_contents(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.rglob("*"):
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            _link_or_copy_file(src, dst)


def _find_mask(process_dir: Path, slide_stem: str) -> Path:
    mask_dir = process_dir / "mask_pic"
    candidates = sorted(mask_dir.glob(f"{slide_stem}*mask*.png"))
    if not candidates:
        candidates = sorted(mask_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No mask PNG found in {mask_dir}")
    return candidates[0]


def _find_summary(region_dir: Path, slide_stem: str) -> Path:
    path = region_dir / f"{slide_stem}_summary.png"
    if path.exists():
        return path
    candidates = sorted(region_dir.glob("*_summary.png"))
    if not candidates:
        raise FileNotFoundError(f"No summary image found in {region_dir}")
    return candidates[0]


def _mask_command(image_dir: Path, process_dir: Path, file_type: str) -> list[str]:
    if file_type == "czi":
        return [
            sys.executable,
            str(APP_DIR / "process_czi.py"),
            "-pic_path", str(image_dir),
            "-out_dir", str(process_dir),
            "-file_type", "czi",
            "-min_area", str(DEFAULT_MIN_AREA),
            "-min_hole", str(DEFAULT_MIN_HOLE),
            "-czi_level", str(CZI_PREVIEW_LEVEL),
            "-black_threshold", "10",
            "-min_black_area", "100",
            "-pen_black_threshold", "110",
            "-pen_min_pixels", "100",
            "-kernel_size", "20",
            "-dilate_iterations", "1",
            "-slice_brightness", "20",
            "-min_fragment_ratio", "0.03",
        ]

    return [
        sys.executable,
        str(APP_DIR / "process.py"),
        "-pic_path", str(image_dir),
        "-out_dir", str(process_dir),
        "-file_type", file_type,
        "-min_area", str(DEFAULT_MIN_AREA),
        "-min_hole", str(DEFAULT_MIN_HOLE),
    ]


def _feature_command(
    image_dir: Path,
    process_dir: Path,
    feature_dir: Path,
    file_type: str,
) -> list[str]:
    if file_type == "czi":
        return [
            sys.executable,
            str(APP_DIR / "cut_norm_feature_czi.py"),
            "-input_folder", str(image_dir),
            "-mask_folder", str(process_dir / "mask_pic"),
            "-output_folder", str(feature_dir),
            "-patch_size", str(CZI_PATCH_SIZE),
            "-read_size", str(CZI_READ_SIZE),
            "-file_type", "czi",
            "-r", str(DEFAULT_REFERENCE_PATCH),
            "-m", "none",
            "--batch_size", str(CZI_BATCH_SIZE),
            "--checkpoint_path", str(DEFAULT_CHECKPOINT_PATH),
            "--num_workers", str(CZI_NUM_WORKERS),
            "--czi_strip_patches", str(CZI_STRIP_PATCHES),
            "-blank_threshold", str(CZI_BLANK_THRESHOLD),
            "--min_mask_ratio", str(CZI_MIN_MASK_RATIO),
        ]

    return [
        sys.executable,
        str(APP_DIR / "cut_norm_feature_copy.py"),
        "-input_folder", str(image_dir),
        "-mask_folder", str(process_dir / "mask_pic"),
        "-output_folder", str(feature_dir),
        "-patch_size", str(DEFAULT_PATCH_SIZE),
        "-read_size", str(DEFAULT_READ_SIZE),
        "-level", str(DEFAULT_LEVEL),
        "-file_type", file_type,
        "-r", str(DEFAULT_REFERENCE_PATCH),
        "-m", "macenko",
        "--batch_size", str(DEFAULT_BATCH_SIZE),
        "--checkpoint_path", str(DEFAULT_CHECKPOINT_PATH),
        "--num_workers", str(DEFAULT_NUM_WORKERS),
        "-blank_threshold", str(DEFAULT_BLANK_THRESHOLD),
    ]


def _analyze_thread(job_id: str):
    job = _job(job_id)
    job_dir = Path(job["job_dir"])
    image_dir = job_dir / "image"
    process_dir = job_dir / "process"
    feature_dir = job_dir / "feature_conch2"
    text_dir = job_dir / "text_features"
    region_dir = job_dir / "region_maps"
    slide_path = Path(job["slide_path"])
    slide_stem = slide_path.stem
    file_type = job["file_type"]
    slide_cache_key = _cache_key(_slide_cache_payload(slide_path, file_type))
    mask_cache_dir = CACHE_DIR / "masks" / slide_cache_key
    feature_cache_dir = CACHE_DIR / "features" / slide_cache_key
    text_cache_key = _cache_key(_text_cache_payload())
    text_cache_dir = CACHE_DIR / "text_features" / text_cache_key

    try:
        _update_job(job_id, status="running", stage="process", progress=0.10)
        _require_file(DEFAULT_REFERENCE_PATCH, "REFERENCE_PATCH_PATH", "Reference patch image")
        _require_file(DEFAULT_CHECKPOINT_PATH, "CONCH_CHECKPOINT_PATH", "CONCH checkpoint")

        if (mask_cache_dir / "mask_pic").exists():
            _append_log(job_id, f"\nCache hit: tissue mask {mask_cache_dir}\n")
            _copy_tree_contents(mask_cache_dir, process_dir)
        else:
            _run_step(
                job_id,
                "Tissue mask",
                _mask_command(image_dir, process_dir, file_type),
                APP_DIR,
            )
            _copy_tree_contents(process_dir, mask_cache_dir)

        _update_job(job_id, stage="features", progress=0.30)
        cached_h5 = feature_cache_dir / f"{slide_stem}.h5"
        if cached_h5.exists():
            _append_log(job_id, f"\nCache hit: patch features {cached_h5}\n")
            _copy_tree_contents(feature_cache_dir, feature_dir)
        else:
            _run_step(
                job_id,
                "Patch feature extraction",
                _feature_command(image_dir, process_dir, feature_dir, file_type),
                APP_DIR,
            )
            _copy_tree_contents(feature_dir, feature_cache_dir)

        _update_job(job_id, stage="text", progress=0.70)
        if (text_cache_dir / "class_features.npy").exists():
            _append_log(job_id, f"\nCache hit: text features {text_cache_dir}\n")
            _copy_tree_contents(text_cache_dir, text_dir)
        else:
            _run_step(
                job_id,
                "Text feature encoding",
                [
                    sys.executable,
                    str(APP_DIR / "encode_text_queries.py"),
                    "--checkpoint_path",
                    str(DEFAULT_CHECKPOINT_PATH),
                    "--output_dir",
                    str(text_dir),
                ],
                APP_DIR,
            )
            _copy_tree_contents(text_dir, text_cache_dir)

        _update_job(job_id, stage="regions", progress=0.82)
        _run_step(
            job_id,
            "Softmax region map",
            [
                sys.executable,
                str(APP_DIR / "retrieve_patches.py"),
                "--text_feature_dir",
                str(text_dir),
                "--h5_dir",
                str(feature_dir),
                "--output_dir",
                str(region_dir),
                "--slide_folder",
                str(image_dir),
                "--file_type",
                file_type,
                "--level",
                str(DEFAULT_LEVEL),
                "--read_size",
                str(CZI_READ_SIZE if file_type == "czi" else DEFAULT_READ_SIZE),
                "--logit_scale",
                str(DEFAULT_LOGIT_SCALE),
                "--max_side",
                str(DEFAULT_MAX_SIDE),
            ],
            APP_DIR,
        )

        summary_path = _find_summary(region_dir, slide_stem)
        _update_job(
            job_id,
            status="complete",
            stage="complete",
            progress=1.0,
            summary_url=_job_url(summary_path),
            csv_url=_job_url(region_dir / f"{slide_stem}_region_labels.csv"),
            distribution_url=_job_url(region_dir / f"{slide_stem}_class_distribution.txt"),
        )
    except Exception as exc:
        _append_log(job_id, f"\nERROR: {exc}\n")
        _update_job(job_id, status="error", stage="error", error=str(exc))


def _create_job_for_slide(slide_source: Path, file_type: str, copy_file: bool) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    image_dir = job_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)

    slide_path = image_dir / slide_source.name
    if copy_file:
        shutil.copy2(slide_source, slide_path)
    else:
        try:
            os.symlink(slide_source, slide_path)
        except OSError:
            shutil.copy2(slide_source, slide_path)

    thumb_path = job_dir / f"{slide_path.stem}_thumbnail.png"
    _make_thumbnail(slide_path, thumb_path)

    job = {
        "job_id": job_id,
        "status": "ready",
        "stage": "ready",
        "progress": 0.0,
        "job_dir": str(job_dir),
        "slide_path": str(slide_path),
        "slide_name": slide_path.name,
        "file_type": file_type,
        "thumbnail_url": _job_url(thumb_path),
        "log": "",
        "created_at": time.time(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    file_type = _safe_ext(file.filename or "")
    tmp_dir = JOBS_DIR / "_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}_{Path(file.filename or 'slide').name}"
    with open(tmp_path, "wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    job = _create_job_for_slide(tmp_path, file_type, copy_file=True)
    return JSONResponse(job)


@app.post("/api/open-path")
def open_path(path: str = Form(...)):
    slide_source = Path(path).expanduser().resolve()
    if not slide_source.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {slide_source}")
    file_type = _safe_ext(slide_source.name)
    job = _create_job_for_slide(slide_source, file_type, copy_file=False)
    return JSONResponse(job)


@app.post("/api/analyze/{job_id}")
def analyze(job_id: str):
    job = _job(job_id)
    if job["status"] == "running":
        return JSONResponse(job)
    _update_job(job_id, status="queued", stage="queued", progress=0.01, error=None)
    thread = threading.Thread(target=_analyze_thread, args=(job_id,), daemon=True)
    thread.start()
    return JSONResponse(_job(job_id))


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    return JSONResponse(_job(job_id))


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WSI Region Analyzer</title>
  <style>
    :root {
      --ink: #20272e;
      --muted: #62707b;
      --line: #d7ded8;
      --green: #2f8065;
      --blue: #6e88bd;
      --bg: #f5f7f4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #fff 0%, var(--bg) 100%);
    }
    main { padding: 34px; max-width: 1720px; margin: 0 auto; }
    .eyebrow {
      color: #516979;
      font-size: 15px;
      font-weight: 800;
      letter-spacing: .13em;
      text-transform: uppercase;
    }
    h1 { margin: 10px 0 6px; font-size: 42px; line-height: 1.05; }
    .sub { color: var(--muted); font-size: 19px; margin-bottom: 26px; }
    .panel {
      border: 1px solid var(--line);
      background: rgba(255,255,255,.88);
      border-radius: 8px;
      padding: 24px;
      box-shadow: 0 12px 35px rgba(31,44,36,.08);
      margin-bottom: 22px;
    }
    .drop {
      border: 2px dashed #9fb6c1;
      border-radius: 8px;
      min-height: 190px;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 24px;
      background: #fbfcfb;
    }
    .drop h2 { margin: 0 0 10px; font-size: 24px; }
    .drop p { margin: 0 0 18px; color: #6a6a6a; }
    input[type="file"] { display: none; }
    .btn, label.btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 48px;
      padding: 0 22px;
      border-radius: 7px;
      border: 1px solid #bcc8c2;
      background: white;
      color: var(--ink);
      font-weight: 750;
      cursor: pointer;
      font-size: 16px;
    }
    .btn.primary { background: var(--blue); border-color: var(--blue); color: white; }
    .btn.green { background: var(--green); border-color: var(--green); color: white; }
    .btn:disabled { opacity: .55; cursor: default; }
    .row { display: grid; grid-template-columns: 1fr auto; gap: 16px; align-items: end; margin-top: 18px; }
    .field label { display: block; margin-bottom: 8px; font-weight: 650; }
    .field input, .field textarea {
      width: 100%;
      min-height: 48px;
      padding: 12px 14px;
      border: 1px solid #bcc8c2;
      border-radius: 7px;
      font-size: 16px;
      background: white;
      font: inherit;
      resize: vertical;
    }
    .field textarea {
      min-height: 96px;
      line-height: 1.45;
    }
    .filename { color: #37654f; margin-top: 15px; font-weight: 750; }
    .bar { height: 16px; border-radius: 99px; background: #dde5df; overflow: hidden; margin: 16px 0 8px; }
    .fill { height: 100%; width: 0%; background: var(--green); transition: width .25s ease; }
    .status { color: var(--muted); font-size: 16px; }
    .results {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 18px;
    }
    .preview {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      display: block;
    }
    .image-scroll {
      width: 100%;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }
    .summary-image {
      width: 100%;
      max-width: 100%;
      height: auto;
      border: 0;
      border-radius: 0;
      background: white;
      display: block;
    }
    .actions { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 18px; }
    .distribution {
      margin-top: 20px;
      display: grid;
      gap: 10px;
      max-width: 860px;
    }
    .dist-row {
      display: grid;
      grid-template-columns: 190px minmax(120px, 1fr) 70px;
      align-items: center;
      gap: 12px;
      font-size: 15px;
    }
    .dist-label {
      display: flex;
      align-items: center;
      gap: 9px;
      min-width: 0;
    }
    .swatch {
      width: 16px;
      height: 16px;
      border: 1px solid rgba(0,0,0,.35);
      flex: 0 0 auto;
    }
    .dist-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .dist-track {
      height: 22px;
      border-radius: 5px;
      background: #e6ece8;
      overflow: hidden;
      border: 1px solid #d2ddd6;
    }
    .dist-bar {
      height: 100%;
      width: 0%;
      background: var(--green);
    }
    .dist-value {
      font-variant-numeric: tabular-nums;
      text-align: right;
      color: #34424a;
      font-weight: 750;
    }
    @media (max-width: 760px) {
      .dist-row {
        grid-template-columns: 1fr 58px;
      }
      .dist-track {
        grid-column: 1 / -1;
      }
    }
    pre {
      white-space: pre-wrap;
      max-height: 320px;
      overflow: auto;
      background: #172026;
      color: #e7f1ea;
      padding: 16px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.45;
    }
    .hidden { display: none; }
  </style>
</head>
<body>
<main>
  <div class="eyebrow">Digital Pathology</div>
  <h1>WSI Region Analyzer</h1>
  <div class="sub">Upload or open a local WSI, preview it, then run region segmentation with CONCH features and softmax region labeling.</div>

  <section class="panel">
    <div class="drop" id="drop">
      <div>
        <h2>Choose a WSI file or drag it here</h2>
        <p>Supported formats: SVS, NDPI, TIFF, CZI</p>
        <label class="btn" for="file">Choose file</label>
        <input id="file" type="file" accept=".svs,.ndpi,.tif,.tiff,.czi" />
        <div id="filename" class="filename"></div>
      </div>
    </div>
    <div class="row">
      <div class="field">
        <label for="serverPath">Open server-side WSI path</label>
        <input id="serverPath" placeholder="/path/to/example.svs" />
      </div>
      <button class="btn" id="openPath">Open path</button>
    </div>
    <div class="bar"><div class="fill" id="uploadFill"></div></div>
    <div class="status" id="uploadStatus">Ready</div>
  </section>

  <section class="panel hidden" id="thumbPanel">
    <h2>Thumbnail Preview</h2>
    <img class="preview" id="thumbnail" />
    <div class="actions">
      <button class="btn green" id="analyze">Run region segmentation</button>
    </div>
  </section>

  <section class="panel hidden" id="progressPanel">
    <h2>Region Segmentation Progress</h2>
    <div class="bar"><div class="fill" id="analysisFill"></div></div>
    <div class="status" id="analysisStatus">Waiting</div>
    <pre id="log"></pre>
  </section>

  <section class="panel hidden" id="resultPanel">
    <h2>Summary</h2>
    <div class="image-scroll">
      <img class="summary-image" id="summary" />
    </div>
    <h3>Class Distribution</h3>
    <div class="distribution" id="distributionChart"></div>
    <div class="actions">
      <a class="btn" id="csvLink" target="_blank">Region CSV</a>
      <a class="btn" id="distLink" target="_blank">Class Distribution</a>
    </div>
  </section>

</main>

<script>
let jobId = null;
let pollTimer = null;

const file = document.getElementById("file");
const drop = document.getElementById("drop");
const filename = document.getElementById("filename");
const uploadFill = document.getElementById("uploadFill");
const uploadStatus = document.getElementById("uploadStatus");
const thumbPanel = document.getElementById("thumbPanel");
const thumbnail = document.getElementById("thumbnail");
const analyzeBtn = document.getElementById("analyze");
const progressPanel = document.getElementById("progressPanel");
const analysisFill = document.getElementById("analysisFill");
const analysisStatus = document.getElementById("analysisStatus");
const log = document.getElementById("log");
const resultPanel = document.getElementById("resultPanel");
const summary = document.getElementById("summary");
const csvLink = document.getElementById("csvLink");
const distLink = document.getElementById("distLink");
const distributionChart = document.getElementById("distributionChart");

const classColors = {
  "malignant tissue": "#d62728",
  "benign tissue": "#377eb8",
  "stroma": "#fdbf6f",
  "lymphocytes": "#63bd7b",
  "necrosis": "#8b008b",
  "adipose tissue": "#efe8c5",
  "tissue artifact": "#7f7f7f",
  "blood vessel": "#f4a6b8",
  "extracellular mucin": "#12dce8",
  "nerve": "#4b148c",
  "hemorrhage": "#a52a2a",
  "smooth muscle": "#d2691e",
  "plasma cells": "#ff1493"
};

function setUploadProgress(pct, text) {
  uploadFill.style.width = `${pct}%`;
  uploadStatus.textContent = text;
}

function showJob(job) {
  jobId = job.job_id;
  thumbPanel.classList.remove("hidden");
  thumbnail.src = job.thumbnail_url + `?t=${Date.now()}`;
  setUploadProgress(100, `Ready: ${job.slide_name}`);
}

file.addEventListener("change", () => {
  const selected = file.files[0];
  if (!selected) return;
  filename.textContent = `${selected.name} · ${Math.round(selected.size / 1024 / 1024)} MB`;
  uploadFile(selected);
});

drop.addEventListener("dragover", (event) => {
  event.preventDefault();
});
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  const selected = event.dataTransfer.files[0];
  if (!selected) return;
  filename.textContent = `${selected.name} · ${Math.round(selected.size / 1024 / 1024)} MB`;
  uploadFile(selected);
});

function uploadFile(selected) {
  const form = new FormData();
  form.append("file", selected);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) {
      const pct = Math.round(event.loaded / event.total * 100);
      setUploadProgress(pct, `Uploading ${pct}%`);
    }
  };
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      showJob(JSON.parse(xhr.responseText));
    } else {
      setUploadProgress(0, xhr.responseText);
    }
  };
  xhr.onerror = () => setUploadProgress(0, "Upload failed");
  setUploadProgress(1, "Uploading...");
  xhr.send(form);
}

document.getElementById("openPath").addEventListener("click", async () => {
  const path = document.getElementById("serverPath").value.trim();
  if (!path) return;
  setUploadProgress(20, "Opening server path...");
  const form = new FormData();
  form.append("path", path);
  const res = await fetch("/api/open-path", { method: "POST", body: form });
  if (!res.ok) {
    setUploadProgress(0, await res.text());
    return;
  }
  showJob(await res.json());
});

analyzeBtn.addEventListener("click", async () => {
  if (!jobId) return;
  analyzeBtn.disabled = true;
  progressPanel.classList.remove("hidden");
  resultPanel.classList.add("hidden");
  await fetch(`/api/analyze/${jobId}`, { method: "POST" });
  poll();
  pollTimer = setInterval(poll, 1800);
});

async function poll() {
  const res = await fetch(`/api/job/${jobId}`);
  const job = await res.json();
  const pct = Math.round((job.progress || 0) * 100);
  analysisFill.style.width = `${pct}%`;
  analysisStatus.textContent = `${job.stage || job.status} · ${pct}%`;
  log.textContent = job.log || "";
  log.scrollTop = log.scrollHeight;

  if (job.status === "complete") {
    clearInterval(pollTimer);
    analyzeBtn.disabled = false;
    resultPanel.classList.remove("hidden");
    summary.src = job.summary_url + `?t=${Date.now()}`;
    csvLink.href = job.csv_url;
    distLink.href = job.distribution_url;
    renderDistribution(job.distribution_url);
    analysisStatus.textContent = "Complete";
  }
  if (job.status === "error") {
    clearInterval(pollTimer);
    analyzeBtn.disabled = false;
    analysisStatus.textContent = `Error: ${job.error || "unknown error"}`;
  }
}

async function renderDistribution(url) {
  distributionChart.textContent = "Loading distribution...";
  try {
    const res = await fetch(url + `?t=${Date.now()}`);
    if (!res.ok) {
      distributionChart.textContent = "Could not load class distribution.";
      return;
    }
    const text = await res.text();
    const rows = parseDistribution(text);
    if (!rows.length) {
      distributionChart.textContent = "No class distribution data found.";
      return;
    }
    distributionChart.innerHTML = rows.map((row) => {
      const color = classColors[row.label] || "#6e88bd";
      const pct = (row.fraction * 100).toFixed(1);
      const width = Math.max(row.fraction * 100, row.fraction > 0 ? 1.5 : 0);
      return `
        <div class="dist-row">
          <div class="dist-label">
            <span class="swatch" style="background:${color}"></span>
            <span class="dist-name" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</span>
          </div>
          <div class="dist-track">
            <div class="dist-bar" style="width:${width}%; background:${color}"></div>
          </div>
          <div class="dist-value">${pct}%</div>
        </div>`;
    }).join("");
  } catch (error) {
    distributionChart.textContent = `Could not render distribution: ${error}`;
  }
}

function parseDistribution(text) {
  return text.trim().split(/\r?\n/).map((line) => {
    const parts = line.trim().split(/\t+/);
    if (parts.length < 3 || parts[0] === "total_patches") return null;
    const count = Number(parts[1]);
    const fraction = Number(parts[2]);
    if (!Number.isFinite(count) || !Number.isFinite(fraction)) return null;
    return { label: parts[0], count, fraction };
  }).filter(Boolean);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
