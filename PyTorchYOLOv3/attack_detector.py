from __future__ import division
import sys
import torch
import torch.nn.functional as F


from PyTorchYOLOv3.models import *
from PyTorchYOLOv3.utils.utils import *
from PyTorchYOLOv3.utils.datasets import *
 

class MyDetectorYolov3():
    def __init__(self,cfgfile=None,weightfile=None):
        if cfgfile==None or weightfile==None:
            print('need configfile or weightfile')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=Darknet(cfgfile).to(self.device)
        self.model.load_state_dict(torch.load(weightfile))
        # self.model.load_darknet_weights(weightfile)
        print('Loading Yolov3 weights from %s... Done!' % (weightfile))

    def attack(self,input_imgs,attack_id=[0],total_cls=85,object_thres=0.1,clear_imgs=None,compare_imgs=None,img_size=None):
        attack_id=[i+5 for i in attack_id]
        input_imgs = F.interpolate(input_imgs, size=img_size).to(self.device)
        self.model.eval()
        detections = self.model(input_imgs) 

        batch=detections.shape[0]
        boxes =detections.shape[1]
        assert total_cls+5==detections.shape[2]

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
