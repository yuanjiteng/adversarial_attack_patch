"""
这个文件用来测试集成之后各个模型的使用是否能够正常运行,不产生包冲突
"""
import numpy as np
import os
import sys
from sqlalchemy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time
from tqdm import tqdm
from torch import autograd
from ensemble_tool.utils import *
from ensemble_tool.model import  TotalVariation
import random

from PyTorchYOLOv3.attack_detector import MyDetectorYolov3 
# from PyTorchYOLOv5.attack_detector import MyDetectorYoLov5
from PyTorchYOLOv7.attack_detector import MyDetectorYoLov7 

# 为什么引用的就是yolov5中的common的

from load_data import InriaDataset, PatchTransformer, PatchApplier
from pathlib import Path
from ipdb import set_trace as st
import argparse
import sys


if __name__ =="__main__":
    print('\n')
    print('import',sys.path)
    cfgfile1 = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv3/config/yolov3.cfg'
    weightfile1='/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv3/weights/yolov3.weights'
    detector1=MyDetectorYolov3(cfgfile1,weightfile1)
    # cfgfile2 = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/data/coco128.yaml'
    # weightfile2='/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/yolov5s.pt'
    # detector2=MyDetectorYoLov5(cfgfile2,weightfile2)

    weightfile3='/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv7/yolov7.pt'
    detector3=MyDetectorYoLov7(weightfile=weightfile3)

    # 其实所有的yolo都可以使用yolov7 这一套来使用吧，主要也就用了1个 attempt_load 的函数？行吧

    print('\n')
    print('exit',sys.path)

