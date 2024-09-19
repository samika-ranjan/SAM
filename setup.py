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
