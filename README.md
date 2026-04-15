# FACDNet: Frequency-Aware Change Detection Network

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of **FACDNet**, a high-performance Siamese network for remote sensing image change detection. It is designed to tackle the challenges of pseudo-changes (e.g., seasonal variations and illumination shifts) and blurred object boundaries through frequency-domain analysis and cross-layer semantic aggregation.

---

## 🌟 Core Innovations & Modules

FACDNet achieves state-of-the-art performance by integrating the following components:

* **Wavelet Interaction Block (WIB):** Unlike traditional spatial attention, WIB utilizes the Haar Discrete Wavelet Transform (DWT) to decompose features into frequency sub-bands. It employs a **Heterogeneous Attention** strategy: Channel Attention for low-frequency global semantics (LL) and Spatial Attention for high-frequency edge textures (LH, HL, HH).
* **Cross-Layer Frequency Context Aggregator (CLFCA):** Implements a top-down semantic injection mechanism. By aligning and upsampling deep-layer low-frequency semantics, it guides and purifies shallow-layer high-frequency boundary features, effectively distinguishing true changes from background noise.
* **Context-guided Difference Fusion Module (CDFM):** A direction-aware fusion module that calculates bidirectional differences. It uses the refined frequency context from CLFCA as a "gating signal" (via Sigmoid activation) to suppress pseudo-change noise before final feature fusion.
* **Joint Focal-Dice Loss:** A custom loss function optimized for the extreme class imbalance (changed vs. unchanged pixels) typical in remote sensing tasks. It forces the model to focus on hard samples and optimizes boundary precision.

---

## 🛠️ Requirements & Environment Setup

This project was developed and tested on **Ubuntu 22.04** with **CUDA 12.1** and **PyTorch 2.5.1**.

### Quick Installation
```bash
git clone [https://github.com/your-username/FACDNet.git](https://github.com/your-username/FACDNet.git)
cd FACDNet
pip install -r requirements.txt
Full Dependencies (requirements.txt)
The following environment configuration was used for our experiments:

Plaintext
albucore==0.0.24
albumentations==2.0.8
annotated-types==0.7.0
anyio==4.12.0
certifi==2025.11.12
click==8.1.8
contourpy==1.3.0
cycler==0.12.1
eval_type_backport==0.3.0
exceptiongroup==1.3.1
filelock==3.19.1
fonttools==4.60.1
fsspec==2025.9.0
h11==0.16.0
hf-xet==1.2.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.1.7
idna==3.11
ImageIO==2.37.2
importlib_resources==6.5.2
Jinja2==3.1.6
joblib==1.5.2
kiwisolver==1.4.7
lazy_loader==0.4
lightning-utilities==0.15.2
MarkupSafe==2.1.5
matplotlib==3.9.4
mpmath==1.3.0
networkx==3.2.1
numpy==2.0.2
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvtx-cu12==12.1.105
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
packaging==25.0
pillow==11.3.0
pydantic==2.12.5
pydantic_core==2.41.5
pyparsing==3.2.5
python-dateutil==2.9.0.post0
PyYAML==6.0.3
safetensors==0.7.0
scikit-image==0.24.0
scikit-learn==1.6.1
scipy==1.13.1
shellingham==1.5.4
simsimd==6.5.3
six==1.17.0
stringzilla==4.3.0
sympy==1.13.1
tabulate==0.9.0
threadpoolctl==3.6.0
tifffile==2024.8.30
timm==1.0.22
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchmetrics==1.8.2
torchvision==0.20.1+cu121
tqdm==4.67.1
triton==3.1.0
typer-slim==0.20.0
typing-inspection==0.4.2
typing_extensions==4.15.0
zipp==3.23.0
📂 Dataset Preparation
We support standard datasets such as LEVIR-CD and SHCD. Please organize your data as follows:

Plaintext
Data_root/
├── train/
│   ├── A/          # Images at Time 1 (T1)
│   ├── B/          # Images at Time 2 (T2)
│   └── label/      # Binary masks (0: unchanged, 255: changed)
├── val/
│   ├── A/, B/, label/
└── test/
    ├── A/, B/, label/
Note: The dataloader resizes images to 256x256 and binarizes masks with a threshold of 127.

🚀 Usage Guide
1. Training
Update your local data and saving paths in the config dictionary at the bottom of train.py, then run:

Bash
python train.py
The script automatically saves the best_model.pth (based on validation F1-score) and latest_model.pth, while generating training curves (training_curves.png).

2. Testing & Evaluation
Update the checkpoint and data paths in test.py, then run:

Bash
python test.py
This script outputs a detailed metrics table (Precision, Recall, F1, IoU, Acc) and saves binary prediction maps to the specified directory.

⚠️ Important Precautions
Path Configuration: The code contains hard-coded absolute paths (e.g., /hy-tmp/...) specific to our server environment. You MUST update these paths in train.py and test.py before running the scripts.

Reproducibility: We use set_seed(42) to ensure deterministic behavior. However, minor variations in metrics may still occur across different GPU architectures (e.g., RTX 4090 vs. A100).

VRAM Requirements: The default batch_size=16 requires at least 12GB of VRAM. If you encounter "Out of Memory" errors, please decrease the batch_size in the config.

Data Security: Do not upload original datasets to this repository. Refer to the official sources for data access.

📝 Citation
If you use this code in your research, please cite our paper:

代码段
@article{facdnet2026,
  title={[Your Paper Title Here]},
  author={[Your Name] and [Co-authors]},
  journal={[Journal Name]},
  year={2026}
}
📧 Contact
For any questions or issues, please open a GitHub Issue or contact the corresponding author at: [Your Email Address].
