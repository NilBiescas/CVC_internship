#!/bin/bash

# Install PyTorch, torchvision, torchaudio, and CUDA toolkit
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install DGL for CUDA 11.8
conda install -c dglteam/label/cu118 dgl

# Install additional Python packages with pip
pip install pandas segmentation-models-pytorch scikit-learn seaborn wget torchdata pydantic
