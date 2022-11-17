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
from ensemble_tool.utils import *
from tqdm import tqdm

import time


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class PatchTransformer(nn.Module):
    """
    提供patch的位置变换,将patch对应到相应的给定的位置上去
    输入：攻击patch labels img_size attack_id
    输出：位置变换之后的adv_batch_masked msk_batch
    注意：
    """

    def __init__(self):
        super(PatchTransformer, self).__init__()

    def forward(self, adv_patch, lab_batch, img_size, with_projection=True, attack_id=[4, 7, 80], pad_value=0):
        # 将原来的for循环改成了矩阵操作
        if (with_projection):
            adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)  ##  torch.Size([1, 1, 3, 50, 50])
            adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1,
                                         -1)  ##  torch.Size([4, 5, 3, 300, 300])
            b, f, c, h, w = adv_batch.size()

            # 从lab_batch中提取cls_ids
            cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # torch.Size([8,14, 1])
            cls_ids = cls_ids.view(b * f)

            # 在cls_ids里面选出有效的ids进行投影变换操作(四个),即id=attack_id的patch
            bool_index = torch.ones(cls_ids.size())
            bool_index = bool_index.type(torch.ByteTensor).cuda()
            # for id in attack_id:
            #     bool_index = bool_index & (cls_ids==id)
            valid_index = torch.nonzero(cls_ids, as_tuple=True)[0]

            # 根据有效下标valid_index从adv_batch中选出有效的adv_batch_select
            adv_batch = adv_batch.view(b * f, c, h, w)
            adv_batch_select = torch.index_select(adv_batch, 0, valid_index)

            # lab_batch 解码+选择有效部分
            lab_batch = lab_batch.view(b * f, 9)  # [8*14 8]
            lab_batch = torch.index_select(lab_batch, 0, valid_index)
            lab_batch[:, 1:] = lab_batch[:, 1:] * img_size  # 全部乘上img_size  坐标对应回原图 img_size=640

            w = torch.tensor(adv_patch.size(-2)).float().cuda()
            h = torch.tensor(adv_patch.size(-1)).float().cuda()
            org = torch.FloatTensor([[0, 0],
                                     [w, 0],
                                     [0, h],
                                     [w, h]]).cuda()
            # 批量计算adv_batch_select的投影矩阵(4个3*3的矩阵组成Mat)
            org = org.repeat(valid_index.size(0), 1, 1)
            dst = torch.narrow(lab_batch, 1, 1, 8)
            dst = dst.view(-1, org.size(-2), org.size(-1))
            Mat = tgm.get_perspective_transform(org, dst).float().cuda()

            # 利用Mat将adv_batch_select进行投影变换
            adv_batch_select = tgm.warp_perspective(adv_batch_select, Mat, dsize=(img_size, img_size))

            # adv_batch中的其他patch赋值0
            adv_batch = torch.cuda.FloatTensor(b * f, c, img_size, img_size).fill_(0)

            # 将四个有效的投影变换结果加回到adv_batch中
            adv_batch = adv_batch.index_put((valid_index,), adv_batch_select)

            adv_batch = adv_batch.view(b, f, c, img_size, img_size)  # 改回原来的维度结构
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

        if os.path.getsize(lab_path):
            label = np.loadtxt(lab_path)
        else:
            # label = np.ones([5])
            label = np.ones([13])
            # 1 为非攻击类别
            # label = [84,0,0,0,0,0,0,0,1,1,0,1,1]

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
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
                lab[:, 5::2] = (lab[:, 5::2] * w + padding) / h
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
                lab[:, 6::2] = (lab[:, 6::2] * h + padding) / w
        resize = transforms.Resize((self.imgsize, self.imgsize))  # 同比缩放 lab值不变
        padded_img = resize(padded_img)  # choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        lab = F.pad(lab, (1, 0, 0, 0), value=1)
        if (pad_size > 0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=0)
        elif pad_size == 0:
            padded_lab = lab
        else:
            raise Exception('max_label 长度设置过小.')
        return padded_lab


class DataEnhancer(nn.Module):
    """
    此时输入为已经经过投影变换的batch 图像
    输出为 图像在 随机环境变换之后的图像（雨天 雾天 光照变换 对比度 旋转 缩放（主要应对不同距离拍摄））
    输入：
            beta：控制某种mask的权重,范围[0-1]由弱到强
            posibility：选择某种特效的概率 如 选择第二种特效的概率=(posibility[1]-posibility[0]) 最后一种=(1-posibility[-1])
    """

    def __init__(self, maskRoot, imgsize=640, betas=[0.75, 0.7],posibility=[0.001, 0.5], maskNumPerCategory=30):
        super(DataEnhancer, self).__init__()
        # 雨雾每个类准备的mask数量,默认30
        self.maskNumPerCategory = maskNumPerCategory
        self.imgsize = imgsize
        self.posibility = posibility
        # 因为每张mask在整个训练过程中都要用到，所以一次性将所有的mask都导入内存中 容量:(640*640*3)*30*2/(1024*1024)=70M
        rainMaskDir = os.path.join(maskRoot, "rainMask")
        fogMaskDir = os.path.join(maskRoot, "fogMask")
        rainMaskFiles = [os.path.join(rainMaskDir, file) for file in os.listdir(rainMaskDir)]
        fogMaskFiles = [os.path.join(fogMaskDir, file) for file in os.listdir(fogMaskDir)]
        MaskFiles = []
        MaskFiles.append(rainMaskFiles)
        MaskFiles.append(fogMaskFiles)
        transform = transforms.ToTensor()
        resize = transforms.Resize((self.imgsize, self.imgsize))
        Mask = []
        for i, beta in enumerate(betas):
            subMask = []
            for j, MaskFile in enumerate(MaskFiles[i]):
                if j >= maskNumPerCategory:
                    break
                mask = transform(Image.open(os.path.join(rainMaskDir, MaskFile)).convert('RGB')).unsqueeze(0)
                subMask.append(mask)
            subMask = torch.cat(subMask) * beta
            Mask.append(subMask)
        Mask = torch.cat(Mask)
        Mask = resize(Mask)
        # 最开始的位置增加一张全是0元素的mask,抽到这张mask对应的增强输出是原图
        self.Mask = F.pad(Mask, (0, 0, 0, 0, 0, 0, 1, 0), value=0)

    def forward(self, img_batch):
        """
        功能：为每张img单独添加一个随机增强效果(雨、雾)
        输入：
            img_batch：贴上patch的图片组
        输出：添加了随机效果的img_batch
        注意：posibility的长度等于类别数减一(无特效,雨,雾),暂时固定,后面改成可变长度
        """
        img_num = img_batch.size()[0]
        mask_num = self.Mask.size()[0]
        category = torch.rand(img_num)
        category_ = torch.zeros(category.size())
        indexInOneCategory = torch.rand(img_num) * self.maskNumPerCategory
        # 0 表示不加
        category_[category <= self.posibility[0]] = 0
        # 1 表示加雨
        category_[category > self.posibility[0]] = 1
        # 2 表示加雾
        category_[category > self.posibility[1]] = 2
        # category_ = torch.ones_like(category)+1
        # 例如：category_ = 2 indexInOneCategory=0.51*30=15.3 indexInMask=45
        # 表示从总共61张mask中选中下标为45的mask(也即30张雨的mask中下标为15的mask)
        indexInMask = torch.ceil(((category_ - 1) * self.maskNumPerCategory + indexInOneCategory)).to(int)
        indexInMask = torch.clamp(indexInMask, 0, mask_num)
        selectedMask = torch.index_select(self.Mask, 0, indexInMask).cuda()
        # 矩阵操作比for循环快
        img_batch_enhanced = img_batch * (1 - selectedMask) + selectedMask
        return img_batch_enhanced


if __name__ == '__main__':
    # 进行验证 读取数据 生成patch 投影变换 patch应用 图像增强 todo
    root = r"F:\Public\multiPatch"
    img_dir = os.path.join(root, "images")
    lab_dir = os.path.join(root, "labels")
    msk_dir = r"F:\Public\TankAeroplaneAmoredVehicle\test\mask"
    results_dir = r"F:\ZAY\projects\10-28\adversarial_attack_patch-main\training_patches\view"
    patch_dir = r"F:\Public\multiPatch\arrow.png"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_image_size = 640
    train_batch_size = 1
    train_loader = torch.utils.data.DataLoader(InriaDataset(img_dir=img_dir,
                                                            lab_dir=lab_dir,
                                                            max_lab=5,
                                                            imgsize=train_image_size,
                                                            shuffle=False),
                                               batch_size=train_batch_size,
                                               shuffle=False,
                                               drop_last=False,
                                               num_workers=10)
    train_loader = DeviceDataLoader(train_loader, device)
    adv_patch_cpu = Image.open(patch_dir)
    transTensor = transforms.ToTensor()
    adv_patch_cpu = transTensor(adv_patch_cpu)
    # adv_patch_cpu = torch.rand((3, 50, 50))
    # patch 应用器 增强器 和 TV计算器
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    data_enhancer = DataEnhancer(msk_dir, train_image_size, betas=[0.75, 0.7],posibility=[0.001, 0.5]).cuda()
    iteration_total = len(train_loader)
    torch.cuda.empty_cache()
    transformPIL = transforms.ToPILImage()
    for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Appling patches and weather',
                                                total=iteration_total):
        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()
        adv_patch_cpu = adv_patch_cpu.cuda()
        # T3 = time.clock()
        adv_batch_t = patch_transformer(adv_patch_cpu, lab_batch, train_image_size, with_projection=True)
        # T4 = time.clock()
        # print(f"time for {train_batch_size} patch_transform is {(T4 - T3) * 1000} ms")
        p_img_batch = patch_applier(img_batch, adv_batch_t)
        # T1 = time.clock()
        p_img_batch = data_enhancer(p_img_batch)
        # T2 = time.clock()
        # print(f"time for {train_batch_size} rain/fog is {(T2 - T1) * 1000} ms")
        batch_len = p_img_batch.size(0)
        lis_ = os.listdir(r'F:\Public\multiPatch\images')
        for i, p_img in enumerate(p_img_batch):
            save_image(p_img, os.path.join(results_dir, lis_[i_batch] + ".jpg"))

