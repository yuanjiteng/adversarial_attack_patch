import fnmatch
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

class ColorMap(nn.Module):
    """
    将生成的patch 从打印后的颜色系->打印前的颜色系 用于打印 这一步加上可能会导致内存溢出  
    """
    def __init__(self,color_print_before_file,color_print_after_file,patch_size):
        super(ColorMap,self).__init__()
        self.color_before = self.get_printability_array(color_print_before_file, patch_size)
        self.color_after = self.get_printability_array(color_print_after_file, patch_size)

    def forward(self,adv_patch):
        color_dist = (adv_patch - self.color_after+0.00000000001) # 1 C H W  和 30 3 H W 相减
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.00000000001  
        color_dist = torch.sqrt(color_dist) # 30 h w
        color_dist_prod, min_index = torch.min(color_dist, 0)  # min_index (h,w)
        h,w=min_index.shape
        min_index2=min_index.expand(3,h,w).unsqueeze(0)
        res=torch.gather(self.color_before,0,min_index2) #(1 3 h w)
        return res

    def get_printability_array(self, printability_file, size):
        printability_list = []
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((1, 1), red))
            printability_imgs.append(np.full((1, 1), green))
            printability_imgs.append(np.full((1, 1), blue))
            printability_array.append(printability_imgs)
        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        color_nums,c,h,w=printability_array.shape
        pa = torch.from_numpy(printability_array)
        pa = pa.expand(color_nums,c,size,size).cuda()
        return pa

if __name__=='__main__':
    image = Image.open('/data1/yjt/adversarial_attack/myattack/training_patches/1 patch.png').convert('RGB')
    transform = transforms.ToTensor()
    image = transform(image)
    adv_patch=image.cuda()
    color_map =ColorMap(color_print_before_file='/data1/yjt/adversarial_attack/myattack/color_before_print.txt',\
        color_print_after_file='/data1/yjt/adversarial_attack/myattack/color_after_print.txt',\
        patch_size=300).cuda()
    adv_patch=color_map(adv_patch)
    save_image(adv_patch,'/data1/yjt/adversarial_attack/myattack/training_patches/color_for_print_1.png')
