import numpy as np
import torch
from tqdm import tqdm
from torch import autograd

from imageio import imread, imwrite
from torchvision.utils import make_grid, save_image

from torch.utils.tensorboard import SummaryWriter   
from PyTorchYOLOv3.attack_detector import MyDetectorYolov3
from PyTorchYOLOv5.attack_detector import MyDetectorYoLov5
from PyTorchYOLOv7.attack_detector import MyDetectorYoLov7
from FasterRCNN.attack_detector import MyFastercnn
from SSD.attack_detector import MySSD

from utils_log import Log
from load_data4 import InriaDataset,DeviceDataLoader,PatchApplier,PatchTransformer,NPSCalculator,DifColorQuantization

import argparse
import warnings
warnings.filterwarnings("ignore")
import time


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def train():
    # writer=SummaryWriter('./logs')
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda or cpu
    train_image_size   = 640
    train_batch_size      = 1
    # 数据集 
    train_loader = torch.utils.data.DataLoader(InriaDataset(img_dir='/data1/yjt/mydatasets/attack_datasets/images/',
                                                            lab_dir='/data1/yjt/mydatasets/attack_datasets/labels/',
                                                            max_lab=5,
                                                            imgsize=train_image_size,
                                                            shuffle=True),
                                                batch_size=train_batch_size,
                                                shuffle=True,
                                                num_workers=10)
    train_loader = DeviceDataLoader(train_loader, device)
    # 检测器
    # detectorfasterrcnn = chooseDector('fasterrcnn')
    # detectorssd = chooseDector('ssd')
    # detectoryolov3 = chooseDector('yolov3')
    detectoryolov5 = chooseDector('yolov5')
    # detectoryolov7 = chooseDector('yolov7')
    # patch 生成器
    adv_patch_cpu =generate_patch("random")

    adv_patch_cpu.requires_grad_(True)
    # adv_patch = torch.sigmoid(adv_patch)
    # adv_patch = torch.nn.Parameter(torch.rand(3,10, 10,device = torch.device('cuda')),requires_grad=True)
    # adv_patch_act=torch.sigmoid(adv_patch)
    # adv_patch_act.requires_grad_(True)

    # patch 应用器 增强器 和 TV计算器
    patch_transformer = PatchTransformer().cuda()
    # patch_transformer = PositionTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    # nps_calculator= NPSCalculator(printability_file='/data1/yjt/adversarial_attack/myattack/30values.txt',patch_size=300).cuda()
    dif_color_quantization=DifColorQuantization(printability_file='/data1/yjt/adversarial_attack/myattack/color_after_print.txt',patch_size=300).cuda()
    # color_map =ColorMap(color_print_before_file='/data1/yjt/adversarial_attack/myattack/color_before_print.txt',color_print_after_file='/data1/yjt/adversarial_attack/myattack/color_after_print.txt',patch_size=300).cuda()
    # 定制学习率
    start_lr = 0.01
    iter_max = 100000 #最大训练步数 
    power = 0.9
    iter_batch=0  #迭代步数
    iter_collect=80000 #集体攻击步数

    # lr = start_lr*(1-iter_batch/iter_max)**power 调整策略 

    # 优化器
    optimizer = torch.optim.Adam([adv_patch_cpu], lr=start_lr, betas=(0.5, 0.999), amsgrad=True)

    # log 为什么这里使用一直失败啊
    logfile='Myattack-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    log=Log(logfile)

    
    # 开始训练
    iteration_total = len(train_loader)
    torch.cuda.empty_cache()
    start_epoch=1
    n_epochs=10000
    pr_loss = []

    for epoch in range(start_epoch, n_epochs+1):
        for i,(img_batch, lab_batch) in enumerate(train_loader):
            iter_batch=iter_batch+1
            with autograd.detect_anomaly():
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                # 对于cpu进行更改？
                # 注意 lab_batch 和img_batch 的 requires_grad 都是false 
                adv_patch =adv_patch_cpu.cuda()
                iter_single=0
                adv_patch=dif_color_quantization(adv_patch)
                adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, with_projection=True,attack_id=[4,7,80])
                p_img_batch = patch_applier(img_batch, adv_batch_t)

                # loss_det_fasterrcnn=detectorfasterrcnn.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=512)
                # loss_det_ssd = detectorssd.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=300)
                # loss_det_yolov3 = detectoryolov3.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=416)
                loss_det_yolov5 = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                # loss_det_yolov7 = detectoryolov7.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                loss_det = loss_det_yolov5 
                # loss_nps=nps_calculator(adv_patch)
            
                loss = loss_det
                # 可以选取一个随机种子 然后对于五个损失函数都进行优化？优化一次和五次的区别？     
                        # f'ssd:{loss_det_ssd.detach().cpu().numpy():8.5f} '\
                        # f'yolov3:{loss_det_yolov3.detach().cpu().numpy():8.5f} '\
                        # f'yolov5:{loss_det_yolov5.detach().cpu().numpy():8.5f} '\
                        # f'yolov7:{loss_det_yolov7.detach().cpu().numpy():8.5f} '\
                msg =   f'loss:{loss.detach().cpu().numpy():8.5f} '\
                        f'learining rate: ', optimizer.param_groups[0]['lr'],\
                        f'iter: {iter_batch}'
                
                log.info(msg)

                loss.backward()
                # print(adv_patch_cpu.grad)
                optimizer.step()
                optimizer.param_groups[0]['lr']=start_lr*(1-iter_batch/iter_max)**power 
                adv_patch_cpu.data.clamp_(0,1)
                # print(torch.max(adv_patch_cpu),torch.max(adv_patch))
                # 更新adv_patch_cpu 

                # 集成
                log.info("collect: ")
                iter_single=0
                while loss.item()>0.3 and iter_single<5: 
                    iter_batch=iter_batch+1
                    iter_single=iter_single+1 #while 控制

                    adv_patch =adv_patch_cpu.cuda()
                    adv_patch=dif_color_quantization(adv_patch)
                    adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, with_projection=True,attack_id=[4,7,80])
                    p_img_batch = patch_applier(img_batch, adv_batch_t)
                    
                    # loss_det_ssd = detectorssd.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=300)
                    # loss_det_yolov3 = detectoryolov3.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.25,clear_imgs=img_batch,img_size=416)
                    loss_det_yolov5 = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                    # loss_det_yolov7 = detectoryolov7.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)

                    loss_det = loss_det_yolov5
                    loss =loss_det
                    loss.backward()
                    optimizer.step()
                    optimizer.param_groups[0]['lr']=start_lr*(1-iter_batch/iter_max)**power

                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)
                    
                        # f'ssd:{loss_det_ssd.detach().cpu().numpy():8.5f} '\
                    msg= f'yolov5:{loss_det_yolov5.detach().cpu().numpy():8.5f} '\
                        # f'yolov7:{loss_det_yolov7.detach().cpu().numpy():8.5f} '\
                        f'learining rate: ', optimizer.param_groups[0]['lr'],\
                        f'iter', iter_batch
                    log.info(msg)
                
                
                # 这部分后面重写
                # print("random: ")
                # iter_single=0
                # while iter_single<20 : 
                #     # 使用随机种子 随机挑选一种方法攻击
                #     adv_patch =adv_patch_cpu.cuda()
                #     adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, with_projection=True,attack_id=[4,7,80])
                #     p_img_batch = patch_applier(img_batch, adv_batch_t)
                #     iter_batch=iter_batch+1
                #     iter_single=iter_single+1 
                #     seed = torch.randint(0,4,(1,))
                #     if seed.item()==0 :
                #         loss = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640) 
                #         # loss = detectoryolov3.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.25,clear_imgs=img_batch,img_size=416)
                #     elif seed.item()==1:
                #         loss = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640) 
                #     elif seed.item()==2:
                #         loss = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)   
                #     elif seed.item()==3:
                #         # loss = detectorssd.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=300)
                #         loss = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                    
                #     loss.backward()
                #     optimizer.step()
                #     optimizer.param_groups[0]['lr']=start_lr*(1-iter_batch/iter_max)**power
                #     optimizer.zero_grad()
                #     adv_patch_cpu.data.clamp_(0,1)
                #     print(
                #         f'loss:{loss.item():8.5f} '\
                #         'learining rate: ', optimizer.param_groups[0]['lr'],\
                #         'iter', iter_batch
                #     )

                del adv_batch_t,loss_det, p_img_batch, loss
                torch.cuda.empty_cache()

        optimizer.zero_grad()
        # 保存训练patch 
        print('save generated patch\n')
        save_image(adv_patch,"training_patches/" + str(epoch) + " patch.png")

def evaluate():
    pass

def chooseDector(str):
    """
    输入string 返回检测器,
    每个检测器存在一个attack攻击方法 返回 [Batch_size,attack_id]
    """
    if  str=='yolov3':
        cfgfile = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv3/config/yolov3.cfg'
        weightfile='/data1/yjt/adversarial_attack/myattack_training_models/yolov3_ckpt_50.pth'
        detector=MyDetectorYolov3(cfgfile,weightfile)
        return detector 

    elif str=='yolov5':
        cfgfile = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/models/yolov5s.yaml'
        weightfile='/data1/yjt/adversarial_attack/myattack_training_models/yolov5s-85-enhance-10epoch-dict.pt'
        detector=MyDetectorYoLov5(cfgfile,weightfile)
        return detector
    
    elif str=='yolov7':
        # cfgfile = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv7/cfg/training/yolov7.yaml'
        # weightfile='/data1/yjt/adversarial_attack/myattack_training_models/yolov7-85-enhance.pt'
        # detector =MyDetectorYoLov7(cfgfile,weightfile)
        # return detector
        pass
    
    elif str=='fasterrcnn':
        weightfile = '/data1/yjt/adversarial_attack/myattack_training_models/FasterRCNN-9.pth'
        detector = MyFastercnn(weightfile)
        return detector
    
    elif str=='ssd':
        weightfile = '/data1/yjt/adversarial_attack/myattack_training_models/SSD-30.pth'
        detector = MySSD(weightfile)
        return detector

def generate_patch(str):
    if str == 'gray':
        adv_patch_cpu = torch.full((3,300,300), 0.5)
        return adv_patch_cpu
    elif str == 'random':
        adv_patch_cpu = torch.rand((3,300,300))
        return adv_patch_cpu
    elif str == 'img':
        image = Image.open('./training_patches/60 patch.png').convert('RGB')
        transform = transforms.ToTensor()
        image = transform(image)
        return image
    elif str=="net":
        # 注意如果使用 需要更改优化参数 优化参数也需要同时更新
        adv_patch_cpu = torch.rand((3,300,300))
        adv_patch_cpu = GeneatorPatch(adv_patch)
        pass
    elif str=="gan":
        # 使用gan 生成某一种类型的patch todo
        pass

if __name__ =='__main__':
    train()

