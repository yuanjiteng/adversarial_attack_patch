import os
import sys
# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
# sys.path.append(os.path.abspath(os.path.join(__dir__, '../..'))) #








import torch.nn.functional as F
import numpy as np
import torch
from FasterRCNN.backbone import resnet50_fpn_backbone
from FasterRCNN.network_files import FasterRCNN




class MyFastercnn():
    def __init__(self,weightfile=None):
        if  weightfile==None:
            raise ValueError ('need configfile or weightfile')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
        self.model = FasterRCNN(backbone=backbone).to(self.device)
        self.model.load_state_dict(torch.load(weightfile, map_location='cpu')["model"])
        print('Loading Fastercnn weights from %s... Done!' % (weightfile))

    def attack(self, input_imgs, attack_id=[0], total_cls=85, object_thres=0.1, clear_imgs=None, compare_imgs=None,
               img_size=None):
        attack_id = [4,7,80]                              #z
        bboxes = []
        prof_max_scores = []
        any_max_scores = []
        input_imgs = F.interpolate(input_imgs, size=img_size).to(self.device)
        self.model.eval()
        for input_img in input_imgs:
            with torch.no_grad():
                outputs = self.model([input_img])[0]
            outputs["boxes"][:, 0] = outputs["boxes"][:, 0] / input_img.size()[-2]
            outputs["boxes"][:, 1] = outputs["boxes"][:, 1] / input_img.size()[-1]
            outputs["boxes"][:, 2] = outputs["boxes"][:, 2] / input_img.size()[-2]
            outputs["boxes"][:, 3] = outputs["boxes"][:, 3] / input_img.size()[-1]

            # create bbox with (batch,7). (x1,y1,x2,y2,score,score,class_id)
            batch = outputs["boxes"].size()[0]
            outputs["labels"] = outputs["labels"] - 1  # without class __background__
            bbox = torch.cat((outputs["boxes"], outputs["scores"].resize(batch, 1), outputs["scores"].resize(batch, 1),
                              outputs["labels"].resize(batch, 1)), 1)

            keep = np.in1d(bbox[:, -1].cpu().numpy(), attack_id)
            bbox.requires_grad = True
            any_max_score = torch.max(bbox[:, -2])
            any_max_scores.append(any_max_score)

            bbox = bbox[keep]
            bbox = bbox[(bbox[:, -2] >= object_thres)]
            if (bbox.size()[0] > 0):
                # get max score
                max_score = torch.max(bbox[:, -2])
                # print("max_score : "+str(max_score))

                bboxes.append(bbox)
                prof_max_scores.append(max_score)
            else:
                bboxes.append(torch.tensor([]))
                prof_max_scores.append(torch.tensor(0.0).to(self.device))
        if(len(prof_max_scores) > 0):
            prof_max_scores = torch.stack(prof_max_scores, dim=0)
        else:
            prof_max_scores = torch.stack(any_max_scores, dim=0) * 0.01
            if(input_imgs.is_cuda):
                prof_max_scores = prof_max_scores.cuda()
            else:
                prof_max_scores = prof_max_scores
        if not clear_imgs == None:
            pass
        if not compare_imgs ==None:
            pass
        return torch.mean(prof_max_scores)