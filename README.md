# GroundingDINODemo

## Getting Started

### Installation

Use Anaconda to create a new environment and install the required packages.

setting CUDA_HOME
https://zhuanlan.zhihu.com/p/565649540s

```
conda info --envs

conda create --name groundingdinodemo python=3.10

conda activate groundingdinodemo

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO/

pip install -e .

create weights folder
cd weights
download https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
put the downloaded file in the weights folder
cd ..

cd ..

pip install gradio
```

### Running the Project

```
python app.py
```