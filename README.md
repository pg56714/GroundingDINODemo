# GroundingDINODemo

> **Security:** see [SECURITY.md](SECURITY.md) for known dependency issues.

## Getting Started

### Installation

Use Anaconda to create a new environment and install the required packages.

setting CUDA_HOME
https://zhuanlan.zhihu.com/p/565649540s

```
uv venv

.venv\Scripts\activate

uv pip install -r pyproject.toml
```

PyTorch is installed from the CUDA 12.6 wheel index (`pytorch-cu126` in `pyproject.toml`), which needs an NVIDIA driver that supports CUDA 12.6 or newer.

create weights folder

cd weights

download https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

put the downloaded file in the weights folder

### Running the Project

```
python gradio_app.py
```
