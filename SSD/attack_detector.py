import os
import sys
# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
# sys.path.append(os.path.abspath(os.path.join(__dir__, '../..'))) #








import torch.nn.functional as F
import numpy as np
import torch
from SSD.src import SSD300, Backbone

from SSD import transforms


class MySSD():
    def __init__(self,weightfile=None):
        if  weightfile==None:
            raise ValueError ('need configfile or weightfile')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = Backbone()
        self.model = SSD300(backbone=backbone).to(self.device)
        self.model.load_state_dict(torch.load(weightfile, map_location='cpu')["model"])
        print('Loading SSD weights from %s... Done!' % (weightfile))

    def attack(self, input_imgs, attack_id=[0], total_cls=85, object_thres=0.1, clear_imgs=None, compare_imgs=None,
               img_size=None):
        # attack_id = [4,7,80]
        # attack_id=[i+5 for i in attack_id]                                  #z
        bboxes = []
        prof_max_scores = []
        any_max_scores = []
        input_imgs_resize = F.interpolate(input_imgs, size=img_size).to(self.device)#ssd输入大小规定300*300
        self.model.eval()
        for input_img,ori_img in zip(input_imgs_resize,input_imgs):
            # with torch.no_grad():
            input_img = input_img.unsqueeze(0)
            data_transform = transforms.Compose([transforms.Normalization()])
            input_img,_ = data_transform(input_img)
            predictions = self.model(input_img)[0]

            predict_boxes = predictions[0]
            if (len(predict_boxes)==0):
                bboxes.append(torch.tensor([]))
                prof_max_scores.append(torch.tensor(0.0).to(self.device))
                continue
            predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * ori_img.shape[1]
            predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * ori_img.shape[2]
            predict_classes = predictions[1]
            predict_scores = predictions[2]

            # create bbox with (batch,7). (x1,y1,x2,y2,score,score,class_id)
            batch = predict_boxes.size()[0]
            predict_classes = predict_classes - 1  # without class __background__
            bbox = torch.cat((predict_boxes, predict_scores.resize(batch, 1), predict_scores.resize(batch, 1),
                              predict_classes.resize(batch, 1)), 1)

            keep = np.in1d(bbox[:, -1].detach().cpu().numpy(), attack_id)
            # bbox.requires_grad = True
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
        # return prof_max_scores
        return torch.mean(prof_max_scores)