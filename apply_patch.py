import numpy as np
import torch
from tqdm import tqdm
from torch import autograd
from PIL import Image 
from torchvision import transforms
from imageio import imread, imwrite
from torchvision.utils import make_grid, save_image

from torch.utils.tensorboard import SummaryWriter   
# from PyTorchYOLOv3.attack_detector import MyDetectorYolov3
# from PyTorchYOLOv5.attack_detector import MyDetectorYoLov5
# from PyTorchYOLOv7.attack_detector import MyDetectorYoLov7
# from FasterRCNN.attack_detector import MyFastercnn
# from SSD.attack_detector import MySSD

from load_data4 import InriaDataset,DeviceDataLoader,PatchApplier,PatchTransformer,NPSCalculator,DifColorQuantization

import argparse
import warnings

import time


warnings.filterwarnings("ignore")

"""
此文件用于将现有的patch 贴图到对应的攻击数据上面 
修改参数 train_imgae_size 和 generate_patch的图像

"""

def run():
    writer=SummaryWriter('./logs')
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda or cpu
    train_image_size   = 640
    train_batch_size   = 1
    # 数据集 
    # 在此处加载数据集时候图像的大小应该不止这么大 实际拍摄时候图像会很大 尝试更改一下
    train_loader = torch.utils.data.DataLoader(InriaDataset(img_dir='/data1/yjt/mydatasets/attack_datasets/images/',
                                                            lab_dir='/data1/yjt/mydatasets/attack_datasets/labels/',
                                                            max_lab=5,
                                                            imgsize=train_image_size,
                                                            shuffle=False),
                                                batch_size=train_batch_size,
                                                shuffle=True,
                                                num_workers=10)
    train_loader = DeviceDataLoader(train_loader, device)

    # patch 生成器
    adv_patch_cpu =generate_patch("img")

    adv_patch_cpu.requires_grad_(True)


    # patch 应用器 
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    # dif_color_quantization=DifColorQuantization(printability_file='/data1/yjt/adversarial_attack/myattack/color_after_print.txt',patch_size=300).cuda()
    # color_map =ColorMap(color_print_before_file='/data1/yjt/adversarial_attack/myattack/color_before_print.txt',color_print_after_file='/data1/yjt/adversarial_attack/myattack/color_after_print.txt',patch_size=300).cuda()
    # 定制学习率
    start_lr = 0.01
    iter_max = 100000 #最大训练步数 
    power = 0.9
    iter_batch=0  #迭代步数
    iter_collect=80000 #集体攻击步数
    iter_save=1000 #每过1000次保存一次patch 
    # lr = start_lr*(1-iter_batch/iter_max)**power 调整策略 


    # 开始训练
    iteration_total = len(train_loader)
    torch.cuda.empty_cache()

    for epoch in range(1):
        for i,(img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',total=iteration_total):
            iter_batch=iter_batch+1
            with autograd.detect_anomaly():
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                adv_patch =adv_patch_cpu.cuda()
                adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, with_projection=True,attack_id=[4,7,80])
                p_img_batch = patch_applier(img_batch, adv_batch_t)
                save_image(p_img_batch,"res_imgs/" + str(i) + ".png") #使用torch 的保存方式稍微有点慢 
                del adv_batch_t, p_img_batch
                torch.cuda.empty_cache()
def generate_patch(str):
    if str == 'gray':
        adv_patch_cpu = torch.full((3,300,300), 0.5)
        return adv_patch_cpu
    elif str == 'random':
        adv_patch_cpu = torch.rand((3,300,300))
        return adv_patch_cpu
    elif str == 'img':
        image = Image.open('./training_patches/60 patch.png').convert('RGB')
        transform = transforms.ToTensor()
        image = transform(image)
        return image
    elif str=="net":
        # 注意如果使用 需要更改优化参数 优化参数也需要同时更新
        adv_patch_cpu = torch.rand((3,300,300))
        adv_patch_cpu = GeneatorPatch(adv_patch)
        pass
    elif str=="gan":
        # 使用gan 生成某一种类型的patch todo
        pass


if __name__ =='__main__':
    run()