import fnmatch
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm

class PositionTransformer(nn.Module):
    """
    提供patch的位置变换,将patch对应到相应的给定的位置上去
    输入：攻击patch labels img_size attack_id 
    输出：位置变换之后的adv_batch_masked msk_batch 
    注意：
    """
    def __init__(self):
        super(PositionTransformer,self).__init__()
    
    def get_warpRMY(self,label,img_size, w, h):  # Points [torch.Tensor]
        w = torch.tensor(w).float().cuda()
        h = torch.tensor(h).float().cuda()
        org = torch.FloatTensor([[0, 0],
                                [w, 0],
                                [0, h],
                                [w, h]]).cuda()
        dst = torch.FloatTensor(
            [[label[5], label[6]], [label[7], label[8]], [label[9], label[10]], [label[11], label[12]]]).cuda()
        # dst = torch.FloatTensor(
        #     [[181.7085427, 168.8442211], [287.839196, 168.8442211], [165.6281407, 233.1658291], [286.2311558,236.3819095]]).cuda()
        org = org.unsqueeze(0)
        dst = dst.unsqueeze(0)
        if label[5] == img_size*90 or (dst > img_size).any():
            warpR = tgm.get_perspective_transform(org, org).float().cuda()
        else:
            warpR = tgm.get_perspective_transform(org, dst).float().cuda()

        return warpR
    
    def forward(self, adv_patch, lab_batch, img_size, with_projection=False,attack_id=[4,7,80]):
            # 投影变换 攻击id参数化 todo
            if(with_projection):
                # 初始patch
                adv_patch = adv_patch.unsqueeze(0)#.unsqueeze(0)  ##  torch.Size([1, 1, 3, 300, 300])
                adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)  ##  torch.Size([8, 14, 3, 300, 300])

                cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # torch.Size([8,14, 1])
                cls_mask = cls_ids.expand(-1, -1, 3)  # torch.Size([8, 14, 3])
                cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 1])
                cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))  # torch.Size([8, 14, 3, 300])
                cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([8, 14, 3, 300, 1])
                cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # torch.Size([8, 14, 3, 300, 300])

                msk_batch4 = torch.cuda.FloatTensor(cls_mask.size()).fill_(4) - cls_mask  # torch.Size([8, 14, 3, 300, 300])
                msk_batch7 = torch.cuda.FloatTensor(cls_mask.size()).fill_(7) - cls_mask  # torch.Size([8, 14, 3, 300, 300])
                msk_batch80 = torch.cuda.FloatTensor(cls_mask.size()).fill_(80) - cls_mask  # torch.Size([8, 14, 3, 300, 300])
                msk_batch = msk_batch4*msk_batch7*msk_batch80
                
                # for i in attack_id:
                #     msk_batch_id =  torch.cuda.FloatTensor(cls_mask.size()).fill_(i) - cls_mask  # torch.Size([8, 14, 3, 300, 300])

                msk_batch[msk_batch != 0] = 1
                msk_batch = 1-msk_batch
                # 只要含有 4 7 80 类 msk_batch全部为1 其他(包括补充)为0 

                b, f, c, h, w = adv_batch.size()
                adv_batch = adv_batch.view(b*f, c, h, w)
                msk_batch = msk_batch.view(b*f, c, h, w)
                adv_batch = adv_batch * msk_batch
                # adv_batch 含有 4 7 80 类 为patch 其他为0  # torch.Size([8*14, 3, 300, 300])

                lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)  #[8 14 5+8]
                lab_batch_scaled[:, :, 1:] = lab_batch[:, :, 1:] * img_size   #全部乘上img_size 必须为方形
                b_, f_, c_ = lab_batch_scaled.size() 
                lab_batch_scaled = lab_batch_scaled.view(b_*f_, c_)  #[8*14 5+8]
                adv_batch_projected = [] 

                #投影变换
                for i,(label,adv_patch1) in enumerate(zip(lab_batch_scaled,adv_batch)):
                    
                    mat = self.get_warpRMY(label,img_size, w = adv_patch.size(-2), h = adv_patch.size(-1))
                    # 虽然一次处理一张patch的投影位置，但是这个函数需要输入的维度是 [B,C,W,H]
                    adv_patch1 = adv_patch1.unsqueeze(0)
                    adv_patch1 = tgm.warp_perspective(adv_patch1, mat, dsize = (img_size, img_size))
                    adv_batch_projected.append(adv_patch1)
                
                adv_batch = torch.cat(adv_batch_projected)
                adv_batch = adv_batch.view(b, f, c, img_size, img_size) # [8 ,14,3,416,416] 注意投影变换之后图像大小已经改变
                
                # pad = (img_size - adv_batch.size(-1)) / 2  # (416-300) / 2 = 58
                # mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
                # adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
                # msk_batch = mypad(msk_batch)
                # adv_batch_masked = adv_batch
            return adv_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        # print("img_batch size : "+str(img_batch.size()))  ##  torch.Size([8, 3, 416, 416])
        # print("adv_batch size : "+str(adv_batch.size()))  ##  torch.Size([8, 14, 3, 416, 416])
        advs = torch.unbind(adv_batch, 1)
        # print("advs (np) size : "+str(np.array(advs).shape))  ##  (14,)
        # print("b[0].size      : "+str(b[0].size()))  ##  torch.Size([8, 3, 416, 416])
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch


class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.
    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/
    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.
        # dataset 读取 inria 数据集，注意label的加载方式 
        # 可能稍微有区别 maxlabel代表 一个文件中最大的labels 数量
    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        # fnmatch 用于文件名字的匹配 
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        # 支持png和jpg 
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')

        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            # label = np.ones([5])
            label = np.ones([13])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)

        label = self.pad_lab(label)
        return image, label
    
    def pad_and_scale(self, img, lab):
        """
        Args:
            img:
        Returns:
        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
                lab[:, 5::2] = (lab[:, 5::2]* w + padding )/ h
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
                lab[:, 6::2] = (lab[:, 6::2]*h + padding) / w
        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=90)#改
        else:
            padded_lab = lab
        return padded_lab


class DataEnhance(nn.Module):
    """
    此时输入为已经经过投影变换的batch 图像
    输出为 图像在 随机环境变换之后的图像（雨天 雾天 光照变换 对比度 旋转 缩放（主要应对不同距离拍摄））
    """
    def __init__(self):
        super(PositionTransformer,self).__init__()

    def forward(self,adv_batch_masked, msk_batch):
        pass 


if __name__ == '__main__':
    # 进行验证 读取数据 生成patch 投影变换 patch应用 图像增强 todo 
    pass