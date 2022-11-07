"""
this file is for adversarial attack patch 
"""
import argparse
import os
import platform
import sys
from pathlib import Path
import torch

import torch.nn.functional as F 
import yaml
from PyTorchYOLOv5.models.yolo import Model as yolov5Model


class MyDetectorYoLov5():
    def __init__(self,cfgfile=None,weightfile=None):
        if cfgfile==None or weightfile==None:
            print('need configfile or weightfile')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # nc ch 也放到参数中
        self.model = yolov5Model(cfgfile, ch=3, nc=85, anchors=None).to(self.device)
        ckpt=torch.load(weightfile,map_location='cpu')
        self.model.load_state_dict(ckpt)

    def load2save():
        # 另外一种加载方式 用于上面的方式
        from PyTorchYOLOv5.models.common import DetectMultiBackend
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weightfile='/data1/yjt/adversarial_attack/myattack/MyEnsemble/yolov5s-85.pt'
        cfgfile = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/models/yolov5s.yaml'
        model = DetectMultiBackend(weightfile, device=device, dnn=False, data=cfgfile, fp16=False)
        torch.save(model.state_dict(), './yolov5ss-85.pt')

    # 攻击参数可以外置 
    def attack(self,input_imgs,attack_id=[0],total_cls=85,object_thres=0.1,clear_imgs=None,compare_imgs=None,img_size=None):
        # 注意 yolov5的图像大小可能和其他的不同
        attack_id=[i+5 for i in attack_id]
        # print(attack_id)
        input_imgs = F.interpolate(input_imgs, size=img_size).to(self.device)
        self.model.eval()
        detections = self.model(input_imgs,augment=False, visualize=False) #v5torch.Size([B, 25200, 85])
        detections = detections[0]
        batch=detections.shape[0]
        boxes =detections.shape[1]
        assert total_cls+5==detections.shape[2]
        # detections = detections.view(batch*boxes,(total_cls+5))
        # detections =  detections[detections[:,:,4] >= object_thres] # [B,x ,85] 
        objectness = detections[:,:,4:5] # [B,x,1]
        classness = detections[:,:,5:] #[B,x,80]
        classness = torch.nn.Softmax(dim=2)(classness) #[B,x,80]
        attack_classness =detections[:,:,attack_id] #[B,x,3] assuming tank airplane and armored vehicles 
        confs = torch.mul(attack_classness,objectness) #[B,x,3]
        confs,_ = torch.max(confs,dim=1) #[B,3]
        confs,_ = torch.max(confs,dim=1)#[B]
        # print(confs.shape)
        # if confs.shape[0]>=3:
        #     confs, _= confs.topk(1,dim=0) 
        # elif confs.shape[0]>=1:
        #     confs, _= confs.topk(1,dim=0) 
        # if not clear_imgs == None:
        #     pass 
        # if not compare_imgs ==None:
        #     pass 
        # print(confs)
        return torch.mean(confs)

