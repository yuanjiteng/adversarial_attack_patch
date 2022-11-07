import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import torch.nn.functional as F

from PyTorchYOLOv7.models.yolo import Model as yolov7Model

# 为什么这样加载的不能够反传？
import sys
class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


class MyDetectorYoLov7():
    def __init__(self,cfgfile=None,weightfile=None):
        if cfgfile==None or weightfile==None:
            print('need configfile or weightfile')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with add_path('/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/'):
            ckpt = torch.load(weightfile, map_location=self.device)
            model = yolov7Model(cfgfile, ch=3, nc=85, anchors=None).to(self.device)
            state_dict = ckpt['model'].float().state_dict()
            model.load_state_dict(state_dict, strict=False)
            self.model = model 
        # self.model = yolov7Model(cfgfile, ch=3, nc=85, anchors=None).to(self.device)
        # ckpt=torch.load(weightfile,map_location='cpu')
        # self.model.load_state_dict(ckpt)
        # cpu or gpu?
        # self.model=self.load2save()
    # def load2save(self):
    #     from PyTorchYOLOv7.models.experimental import attempt_load
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     weightfile='/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/yolov7-85.pt'
    #     cfgfile = '/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/cfg/training/yolov7.yaml'
    #     with add_path('/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/'):
    #         ckpt = torch.load(weightfile, map_location=device)
    #         model = yolov7Model(cfgfile, ch=3, nc=85, anchors=None).to(self.device)
    #         state_dict = ckpt['model'].float().state_dict()
    #         model.load_state_dict(state_dict, strict=False)
    #         # 这和上面的加载方式没有区别
    #     # model = attempt_load(weightfile, map_location=device)
    #     print('success load v7')
    #     # torch.save(model.state_dict(), './yolov5ss-85.pt')
    #     return model

    def attack(self,input_imgs,attack_id=[0],total_cls=85,object_thres=0.1,clear_imgs=None,compare_imgs=None,img_size=None):
        # 注意 yolov7 大小为640
        attack_id=[i+5 for i in attack_id]
        # print(attack_id)
        input_imgs = F.interpolate(input_imgs, size=img_size).to(self.device)
        self.model.eval()
        detections = self.model(input_imgs,augment=False) #v7 测试时候 torch.Size([B, 25200 90]) 训练时候:[4,3,80,80,90]
        # 在训练和测试模式下，yolov7的输出是不一样的
        # print(detections[0].shape)
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
        # if not clear_imgs == None:
        #     pass 
        # if not compare_imgs ==None:
        #     pass 
        return torch.mean(confs)