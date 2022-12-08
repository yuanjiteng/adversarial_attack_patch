import numpy as np
import torch
from tqdm import tqdm
from torch import autograd
from ensemble_tool.utils import *
from ensemble_tool.model import  TotalVariation
import cv2

from torch.utils.tensorboard import SummaryWriter   
from PyTorchYOLOv3.attack_detector import MyDetectorYolov3
from PyTorchYOLOv5.attack_detector import MyDetectorYoLov5
from PyTorchYOLOv7.attack_detector import MyDetectorYoLov7
from FasterRCNN.attack_detector import MyFastercnn
from SSD.attack_detector import MySSD
from shutil import copyfile 

# from load_data import InriaDataset, PatchTransformer, PatchApplier
from load_data2 import InriaDataset,PatchApplier,PositionTransformer,DataEnhancer
import argparse
import warnings
warnings.filterwarnings("ignore")

"""
此文件用于数据增强,利用mask image 和 颜色域变换 完成图像的数据增强并且保存
"""


# 读取 
def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


def run():
    img_path='/data1/yjt/mydatasets/images/myval/'
    label_path='/data1/yjt/mydatasets/labels/myval/'
    filenames_imgs=os.listdir(img_path)
    filenames_labels=os.listdir(label_path)
    filenames_imgs.sort()
    filenames_labels.sort()
    img_nums = len(filenames_imgs)
    fog_imgs=os.listdir('/data1/yjt/adversarial_attack/myattack/mask/myFog/')
    fog_nums = len(fog_imgs)
    rain_imgs=os.listdir('/data1/yjt/adversarial_attack/myattack/mask/rainMask/')
    rain_nums=len(rain_imgs)

    save_img_path='/data1/yjt/mydatasets/images/myval_light/'
    save_label_path='/data1/yjt/mydatasets/labels/myval_light/'

    for i in range(img_nums):
        path1= os.path.join(img_path,filenames_imgs[i])
        path2=os.path.join(label_path,filenames_labels[i])
        print(path1,path2)
        img=cv2.imread(path1)
        img =cv2.resize(img,(640,640))
        # 随机读取一张雾图像
        # a=np.random.randint(0, high=fog_nums)
        # print(a)
        # 转为numpy
        # img=np.float(img)/255.0
        # fog=cv2.imread('/data1/yjt/adversarial_attack/myattack/mask/myFog/'+fog_imgs[a])
        # b=np.random.randint(0,rain_nums)
        # print(b)
        # rain = cv2.imread('/data1/yjt/adversarial_attack/myattack/mask/rainMask/'+rain_imgs[b])
        # # res=np.maximum(rain,img)
        # # res=img*(255-0.5*rain)+0.5*rain
        # res = rain*0.25+0.75*img


        # res=np.maximum(fog,img)
        # res=np.clip(res, 0, 255)
        # res=0.8*res+0.2*img
    
        # # 对比度  RGB + (RGB - Threshold) * Contrast contrast的值为(-1-1)
        # contrast=np.random.random(1) #0-1随机数
        # contrast=1*(contrast-0.5) # -0.5~0.5
        # img=img+(img-127)*contrast
        # img=np.uint8(np.clip(img,0,255))
        YCrCb=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

        # 亮度调整 RGB-YcbCr-Y=Y+-
        brightness=int((np.random.random(1)-2)*0.2*255) # -0.4~0.1 
        flag= np.random.random(1)
        if flag>0.5:
            flag=1
        else:
            flag=-1
        brightness=brightness*flag

        print(brightness)
        print(YCrCb.shape)
        YCrCb[:,:,0] =cv2.add(YCrCb[:,:,0],brightness)
        img=cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
        img=np.uint8(np.clip(img,0,255))
        res=img
        
        
        name=filenames_imgs[i][:-4]
        cv2.imwrite(save_img_path+name+'_light'+'.jpg',res)
        copyfile(path2,save_label_path+name+'_light'+'.txt')
        

if __name__ =='__main__':
    run()

#