# UNet_Practice

This repository provides tools for training and evaluating UNet models. 

## Table of Contents

1. [Setup](#setup)
2. [Usage](#usage)

## Setup 

**Note:** This is only for GPU
1. Create Python 3.10 Environment
```bash
conda create -n [environment name] python=3.10 -y
conda activate [environment name]
```
2. Clone this repository.

```bash
sudo apt-get install git
git clone https://github.com/Masato-Hayazaki/UNet_Practice.git

cd [repository name]
```

3. Install required packages.

```bash

# CUDA 12.1
pip install -r requirements_12_1.txt

# CUDA 11.8
pip install -r requirements_11_8.txt
```

## Usage

1. Train and test the model.

```bash
python train_unet.py
```

