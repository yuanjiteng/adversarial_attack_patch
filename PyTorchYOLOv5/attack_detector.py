"""
this file is for adversarial attack patch 
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import torch
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# import os
# print(os.getcwd())
# sys.path.append('PyTorchYOLOv5')
# sys.path.append(os.path.join(os.getcwd(),'PyTorchYOLOv5'))
# print(sys.path)

# ['/data1/yjt/adversarial_attack/myattack', 
# '/home/yjt/anaconda3/envs/torch17/lib/python37.zip',
#  '/home/yjt/anaconda3/envs/torch17/lib/python3.7', 
#  '/home/yjt/anaconda3/envs/torch17/lib/python3.7/lib-dynload', 
#  '/home/yjt/anaconda3/envs/torch17/lib/python3.7/site-packages',
#   '/home/yjt/anaconda3/envs/torch17/lib/python3.7/site-packages/IPython/extensions']


# 如果不sys.path.append('PyTorchYOLOv5/') 那么ModuleNotFoundError: No module named 'modelsv5'


# 和v3的models存在冲突了所以报错 
from PyTorchYOLOv5.models.common import DetectMultiBackend
import torch.nn.functional as F 


class MyDetectorYoLov5():
    def __init__(self,cfgfile=None,weightfile=None):
        if cfgfile==None or weightfile==None:
            print('need configfile or weightfile')
        # print(os.getcwd())
        # print(os.path.dirname(__file__))
        # 这两个是不一样的所以可以进行判断了
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend(weightfile, device=self.device, dnn=False, data=cfgfile, fp16=False)

    def attack(self,input_imgs,attack_id=[0],total_cls=85,object_thres=0.1,clear_imgs=None,compare_imgs=None,img_size=None):
        # 注意 yolov5的图像大小为640 480大小
        for index in attack_id:
            index+=5
        input_imgs = F.interpolate(input_imgs, size=img_size).to(self.device)
        self.model.eval()
        detections = self.model(input_imgs,augment=False, visualize=False) #v5torch.Size([B, 25200, 85])
        detections = detections[0]
        batch=detections.shape[0]
        boxes =detections.shape[1]
        assert total_cls+5==detections.shape[2]
        detections = detections.view(batch*boxes,(total_cls+5))
        detections =  detections[detections[:,4] >= object_thres] # [x ,85] 
        objectness = detections[:,4:5] # [x,1]
        
        classness = detections[:,5:] #[x,80]
        classness = torch.nn.Softmax(dim=1)(classness) #[x,80]
        attack_classness =detections[:,attack_id] #[x,3] assuming tank airplane and armored vehicles 
        confs = torch.mul(attack_classness,objectness) #[x,3]
        # 至少存在一种吗 也不一定 或者不用topk 直接全选
        if confs.shape[0]>=3:
            confs, _= confs.topk(3,dim=0) 
        elif confs.shape[0]>=1:
            confs, _= confs.topk(1,dim=0) 

        if not clear_imgs == None:
            pass 
        if not compare_imgs ==None:
            pass 
        return torch.mean(confs)