import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def setup_environment():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torch.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    
import sys
!{sys.executable} -m pip install opencv-python matplotlib
!{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

