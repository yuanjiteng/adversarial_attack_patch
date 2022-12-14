import numpy as np
import torch
import math
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
from patchnet import PatchNet

from utils_log import Log
from load_data4 import InriaDataset,DeviceDataLoader,PatchApplier,PatchTransformer,NPSCalculator,DifColorQuantization,PatchEnhancer,BatchDataEnhancer

import argparse
import warnings
warnings.filterwarnings("ignore")
import time
"""
采用网络方式生成patch, 增加参数量,
"""

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def train(weightfile_train,weightfile_eval):
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda or cpu
    train_image_size   =640
    train_batch_size      = 8
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
    detectoryolov5 = chooseDector('yolov5')
    cfgfile = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/models/yolov5s.yaml'
    weightfile=weightfile_train
    detectoryolov5=MyDetectorYoLov5(cfgfile,weightfile)

    # patch 生成器
    # adv_patch_cpu =generate_patch("random")
    adv_patch_cpu = torch.rand((1,3,300,300))
    adv_patch_cpu.requires_grad_(True)
   
    # adv_patch_cpu.requires_grad_(True)
    model = PatchNet()
    # model=model.cuda()
    # model.train()

    # patch 应用器 增强器 和 TV计算器
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    dif_color_quantization=DifColorQuantization(printability_file='/data1/yjt/adversarial_attack/myattack/color_after_print.txt',patch_size=300).cuda()
    patch_enhancer = PatchEnhancer().cuda()

    rain_msk_dir='/data1/yjt/mydatasets/mask/rainMask/'
    fog_msk_dir='/data1/yjt/mydatasets/mask/myFog/'
    data_enhancer = BatchDataEnhancer(rain_msk_dir,fog_msk_dir, train_image_size,posibility=[0.25,0.5,0.75, 1]).cuda()
    
    # 定制学习率
    start_lr = 0.001
    iter_max = 30000 #最大训练步数 
    power = 0.9
    iter_batch=0  #迭代步数
    # iter_collect=80000 #集体攻击步数
    # lr = start_lr*(1-iter_batch/iter_max)**power 调整策略 

    # params = [adv_patch_cpu]
    params =model.parameters()

    # 优化器
    optimizer = torch.optim.Adam(params, lr=start_lr, betas=(0.5, 0.999), amsgrad=True)
    logfile='Myattack-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    log=Log(logfile)

    # 开始训练
    iteration_total = len(train_loader)
    torch.cuda.empty_cache()
    start_epoch=0
    n_epochs=math.floor(iter_max/iteration_total)
    print(n_epochs)

    for epoch in range(start_epoch, n_epochs):
        for i,(img_batch, lab_batch) in enumerate(train_loader):
            iter_batch=iter_batch+1
            with autograd.detect_anomaly():
                adv_patch =adv_patch_cpu.cuda()
                print("optim: ")
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                model=model.cuda()
                adv_patch = model(adv_patch)

                adv_patch=dif_color_quantization(adv_patch)
                adv_patch = patch_enhancer(adv_patch)
                adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, with_projection=True,attack_id=[4,7,80])
                p_img_batch = patch_applier(img_batch, adv_batch_t)
                p_img_batch = data_enhancer(p_img_batch)

                loss_det_yolov5 = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
             
            
                loss = loss_det_yolov5

                msg= f"yolov5:{loss.detach().cpu().numpy():8.5f},"\
                     f"learining rate:{optimizer.param_groups[0]['lr']},"\
                     f"iter:{iter_batch},"\
                     f"epoch:{epoch}"
                print(msg)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                optimizer.param_groups[0]['lr']=start_lr*(1-iter_batch/iter_max)**power

                # print('优化之后：')
                # for name, parms in model.named_parameters():
                #     if name=='en0.conv.weight':	
                #         print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                #         ' -->grad_value:',parms.grad,'value:',parms)
                # iter_single=0
                # while loss>0.5 and iter_single<5: 
                #     iter_single=iter_single+1 
                #     adv_patch =adv_patch_cpu.cuda()
                #     model=model.cuda()

                #     adv_patch = model(adv_patch)
                #     # adv_patch=dif_color_quantization(adv_patch)
                #     adv_patch = patch_enhancer(adv_patch)
                    
                #     adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, with_projection=True,attack_id=[4,7,80])
                #     p_img_batch = patch_applier(img_batch, adv_batch_t)
                #     # p_img_batch = data_enhancer(p_img_batch)
                    
                #     loss_det_yolov5 = detectoryolov5.attack(input_imgs=adv_patch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
        
                #     loss_det = loss_det_yolov5
                #     loss =loss_det

                #     loss.backward()
                #     optimizer.step()
                #     optimizer.zero_grad()
                    # adv_patch_cpu.data.clamp_(0,1)
                    # print('优化之后：')
                    # for name, parms in model.named_parameters():
                    #     if name=='en0.conv.weight':	
                    #         print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                    #         ' -->grad_value:',parms.grad,'value:',parms)

                    # msg=f"yolov5:{loss.detach().cpu().numpy():8.5f}," \
                    #     f"learining rate:{optimizer.param_groups[0]['lr']},"\
                    #     f"iter:{iter_batch},"\
                    #     f"epoch:{epoch},"
                    # print(msg)

                del loss
                torch.cuda.empty_cache()

        optimizer.zero_grad()
        # 保存训练patch 
        adv_patch=dif_color_quantization(adv_patch)
        print('save generated patch\n')
        save_image(adv_patch,"training_patches_net/"+str(epoch) + " patch.png")
    
    # evaluate(adv_patch,n_epochs,log,weightfile_eval)

def evaluate(adv_patch,epoch,log,weightfile_eval):
    log.info('begin evaluate the patch...')
    # 第一步，生成adv_patch 对应的结果图像
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    eval_image_size   = 640
    eval_batch_size   = 1
    eval_loader = torch.utils.data.DataLoader(InriaDataset(img_dir='/data1/yjt/mydatasets/attack_datasets/images/',
                                                            lab_dir='/data1/yjt/mydatasets/attack_datasets/labels/',
                                                            max_lab=5,
                                                            imgsize=eval_image_size,
                                                            shuffle=False),
                                                batch_size=eval_batch_size,
                                                shuffle=True,
                                                num_workers=10)
    eval_loader = DeviceDataLoader(eval_loader, device)

    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()

    torch.cuda.empty_cache()
    for epoch in range(1):
        for i,(img_batch, lab_batch) in enumerate(eval_loader):
            with autograd.detect_anomaly():
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                # adv_patch =adv_patch_cpu.cuda()
                adv_batch_t = patch_transformer(adv_patch, lab_batch, eval_image_size, with_projection=True,attack_id=[4,7,80])
                p_img_batch = patch_applier(img_batch, adv_batch_t)
                save_image(p_img_batch,"res_imgs/" + str(i) + ".png")
                print("res_imgs/" + str(i) + ".png")
                del adv_batch_t, p_img_batch
                torch.cuda.empty_cache()
    # 第二步，计算攻击成功率 注意，此时会导致PytorchYOLOv5的目录被包含到主目录
    # 也许会出问题 
    from PyTorchYOLOv5.detect2 import attack_cal
    conf_thres_clean=0.75
    weight_clean= weightfile_eval
    source_clean='/data1/yjt/mydatasets/attack_datasets/images/'
    conf_thres_patch=0.75
    weight_patch= weightfile_eval
    source_patch= '/data1/yjt/adversarial_attack/myattack/res_imgs/'
    successed_predict,failed_predict,successed_predict1,failed_predict1=attack_cal(conf_thres_clean=conf_thres_clean,
            weight_clean= weight_clean,
            source_clean=source_clean,
            conf_thres_patch=conf_thres_patch,
            weight_patch= weight_patch,
            source_patch= source_patch
            )
    log.info(f'攻击patch epoch数目,{epoch}')
    log.info('攻击模型: yolov5')
    log.info(f'置信度设置:,{conf_thres_clean},{conf_thres_patch}')
    log.info(f"干净图像识别成功数目,{successed_predict},干净识别失败数目,{failed_predict}")
    log.info(f"对抗样本识别成功数目,{successed_predict1},对抗样本识别失败数目,{failed_predict1}")
    log.info(f"攻击成功率:,{(successed_predict-successed_predict1)/successed_predict}")

    conf_thres_clean=0.7
    weight_clean= weightfile_eval
    source_clean='/data1/yjt/mydatasets/attack_datasets/images/'
    conf_thres_patch=0.7
    weight_patch= weightfile_eval
    source_patch= '/data1/yjt/adversarial_attack/myattack/res_imgs/'
    successed_predict,failed_predict,successed_predict1,failed_predict1=attack_cal(conf_thres_clean=conf_thres_clean,
            weight_clean= weight_clean,
            source_clean=source_clean,
            conf_thres_patch=conf_thres_patch,
            weight_patch= weight_patch,
            source_patch= source_patch
            )
    log.info(f'攻击patch epoch数目,{epoch}')
    log.info('攻击模型: yolov5')
    log.info(f'置信度设置:,{conf_thres_clean},{conf_thres_patch}')
    log.info(f"干净图像识别成功数目,{successed_predict},干净识别失败数目,{failed_predict}")
    log.info(f"对抗样本识别成功数目,{successed_predict1},对抗样本识别失败数目,{failed_predict1}")
    log.info(f"攻击成功率:,{(successed_predict-successed_predict1)/successed_predict}")

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

    for i in range(9,10,1):
        weightfile_train='/data1/yjt/adversarial_attack/myattack_training_models/yolov5s/epoch'+str(i)+'-dict.pt'
        weightfile_eval ='/data1/yjt/adversarial_attack/myattack_training_models/yolov5s/weights/epoch'+str(i)+'.pt'
        train(weightfile_train,weightfile_eval)

