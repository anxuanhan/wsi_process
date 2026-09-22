# WSI Region Analyzer

WSI Region Analyzer is a CONCH-based whole-slide image analysis tool. Its web interface accepts an uploaded slide or a slide path on the server and automatically:

1. detects tissue and creates a tissue mask;
2. extracts patches, applies Macenko stain normalization, and generates CONCH features;
3. classifies and visualizes histopathology regions;
4. exports a summary image, region-label CSV, and class statistics.

Supported formats are `.svs`, `.ndpi`, `.tif`, `.tiff`, and `.czi`.

## Quick Start with Codex
For users with limited experience in environment setup, the easiest way to install this project is with Codex. 
### 1. Download the project
```
   git clone https://github.com/anxuanhan/wsi_process.git
   cd wsi_process
```
### 2. Ask Codex to install it
Open Codex in the wsi_process directory and use:

```text
Read README.md and follow its instructions to complete the environment checks and project installation.

Create the wsi-process Conda environment with Python 3.10 and install all required packages using the versions specified in the README, including Conda OpenSlide and requirements.txt.

Check the CONCH checkpoint. If it is missing, stop and tell me to download it manually from Hugging Face after accepting the license and authenticating my account.

Verify all dependencies and CUDA availability. Do not change any model or processing parameters.

After verification, start main.py and provide the local URL or the SSH tunnel command needed to access it.
```


## If you prefer to set up the environment and dependencies manually, follow the steps below.


## 1. Requirements

- Linux
- Python 3.10
- An NVIDIA GPU; at least 16 GB of VRAM is recommended


Enter the downloaded project directory:

```bash
   git clone https://github.com/anxuanhan/wsi_process.git
   cd wsi_process
```

## 2. Create the environment

Install the native OpenSlide library with Conda:

```bash
conda create -n wsi-process python=3.10 -y
conda activate wsi-process
conda install -c conda-forge openslide=3.4.1 -y

python -m pip install -r requirements.txt
```

Some clusters do not automatically load shared libraries from the active Conda environment. Set the library path before running the application:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Verify the installation:

```bash
python -c "import numpy, torch, torchvision, cv2, openslide, torchstain, wsi_normalizer; \
print('NumPy:', numpy.__version__); \
print('PyTorch:', torch.__version__); \
print('torchvision:', torchvision.__version__); \
print('OpenCV:', cv2.__version__); \
print('OpenSlide Python:', openslide.__version__); \
print('CUDA available:', torch.cuda.is_available())"
```

This project pins `numpy==1.23.5`. Do not upgrade to NumPy 2.x because PyTorch 1.13 and some compiled extensions may fail to load.

## 3. Download the CONCH checkpoint

The CONCH checkpoint is required. Sign in to Hugging Face, request access, and accept the terms on the model page:

- <https://huggingface.co/MahmoodLab/CONCH>

After access is approved, authenticate in the terminal. Never place a Hugging Face token in this README, source code, or Git history.

```bash
huggingface-cli login
mkdir -p checkpoints/conch

huggingface-cli download MahmoodLab/CONCH pytorch_model.bin \
  --local-dir checkpoints/conch
```

The final checkpoint location should be:

```text
wsi_process/checkpoints/conch/pytorch_model.bin
```

The repository includes `assets/reference_patch.png` for Macenko normalization, so a separate reference patch is not normally required.

Check both required files:

```bash
test -f checkpoints/conch/pytorch_model.bin && echo "CONCH checkpoint: OK"
test -f assets/reference_patch.png && echo "Reference patch: OK"
```



## 4. Start the web application

First confirm that the GPU is available on the current node:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Start the server:

```bash
cd /path/to/wsi_process
conda activate wsi-process
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python main.py
```

The application is ready when this message appears:

```text
Uvicorn running on http://0.0.0.0:8000
```

For a local installation, open:

```text
http://127.0.0.1:8000
```


## 5. Access a remote server

If the application is running on a GPU compute node, create an SSH tunnel from the local computer. In this example, `login.example.edu` is the login node and `gpu-node` is the compute node:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -L 8010:127.0.0.1:8000 \
  -J username@login.example.edu \
  username@gpu-node
```

Keep that terminal open and visit:

```text
http://127.0.0.1:8010
```

If the cluster does not support ProxyJump, let the login node forward directly to the compute node:

```bash
ssh -N -L 8010:gpu-node:8000 username@login.example.edu
```

## 6. Default processing settings

### SVS, NDPI, and TIFF

- WSI level: `0`
- Read area: `448 x 448`
- Model input: `224 x 224`
- Batch size: `32`
- Workers: `2`
- Stain normalization: Macenko

### CZI

- Source read area: `656 x 656`
- Model input: `224 x 224`
- Batch size: `128`
- Workers: `16`
- CZI strip patches: `32`
- Stain normalization: Macenko

On a node with limited resources, reduce `CZI_BATCH_SIZE`, `CZI_NUM_WORKERS`, and `CZI_STRIP_PATCHES` near the top of `main.py`.





## 7. Main files

```text
main.py                    Web server and job orchestration
process.py                 SVS/NDPI/TIFF tissue-mask generation
process_czi.py             CZI tissue-mask generation
cut_norm_feature_copy.py   SVS/NDPI/TIFF patches and CONCH features
cut_norm_feature_czi.py    CZI patches and CONCH features
czi_reader.py              CZI reader adapter
encode_text_queries.py     Histopathology text features
retrieve_patches.py        Region classification and visualization
requirements.txt           Pinned Python dependencies
assets/reference_patch.png Default Macenko reference patch
```

CONCH source code and license information: <https://github.com/mahmoodlab/CONCH>
