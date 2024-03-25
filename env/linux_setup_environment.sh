#!/bin/bash
conda env create -f environment.yml
conda activate alan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118