# Parallel Enhancement Net      

- Parallel Enhancement Net
  - [0. Background](#0-background)
  - [1. Pre-request](#1-pre-request)
    - [1.1. Environment](#11-environment)
    - [1.2. DCNv2](#12-dcnv2)
  - [2. Train](#2-train)
  - [3. Test](#3-test)

 
  


## 0. Background

PyTorch implementation of Parallel Enhancement Net

## 1. Pre-request

### 1.1. Environment

- Ubuntu 20.04/18.04
- CUDA 10.1
- PyTorch 1.6
- Packages: tqdm, lmdb, pyyaml, opencv-python, scikit-image

Suppose that you have installed CUDA 10.1, then:

```bash
git clone --depth=1 https://github.com/RyanXingQL/STDF-PyTorch 
cd STDF-PyTorch/
conda create -n stdf python=3.7 -y
conda activate stdf
python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```

### 1.2. DCNv2

**Build DCNv2.**

```bash
cd ops/dcn/
bash build.sh
```

**(Optional) Check if DCNv2 works.**

```bash
python simple_check.py
```

> The DCNv2 source files here is different from the [open-sourced version](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) due to incompatibility. [[issue]](https://github.com/open-mmlab/mmediting/issues/84#issuecomment-644974315)

### 1.3 change dataset
Modify the make_peak_label.py to creat label file.
Change the path of hdf5 file in /dataset/ChallengeDataset.py 
## 2. Train
Change the path of training data in option_R3_ChallengeDataset.yml.
Run:
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train.py --opt_path option_R3_ChallengeDataset.yml

## 3. Test
Run:
CUDA_VISIBLE_DEVICES=2 python test.py --opt_path option_R3_ChallengeDataset.yml






