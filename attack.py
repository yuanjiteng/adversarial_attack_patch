import numpy as np
import torch
from tqdm import tqdm
from torch import autograd
from ensemble_tool.utils import *
from ensemble_tool.model import  TotalVariation
from imageio import imread, imwrite

from torch.utils.tensorboard import SummaryWriter   
# from PyTorchYOLOv3.attack_detector import MyDetectorYolov3
from PyTorchYOLOv5.attack_detector import MyDetectorYoLov5
from PyTorchYOLOv7.attack_detector import MyDetectorYoLov7
from FasterRCNN.attack_detector import MyFastercnn
from SSD.attack_detector import MySSD

from load_data import InriaDataset, PatchTransformer, PatchApplier
# from pathlib import Path
# from ipdb import set_trace as st
import argparse
import warnings

warnings.filterwarnings("ignore")

def parser_opt(known=False):
    Gparser = argparse.ArgumentParser(description='Advpatch Training')
    # 这里后面改成list,可能对很多model同时进行 
    Gparser.add_argument('--models', default=['yolov5','yolov7'], type=str, help='options: yolov3,yolov5,yolov7,fasterrcnn,ssd')
    Gparser.add_argument('-generator',default='gray',type=str, help='options: gray,rgb')
    Gparser.add_argument('--pathch-size',default='300',type=int,help='patchsize')
    return Gparser.parse_known_args()[0] if known else Gparser.parse_args()

def run(opt):
    writer=SummaryWriter('./logs')
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda or cpu
    train_image_size   = 640
    train_batch_size      = 1
    # 数据集 
    train_loader = torch.utils.data.DataLoader(InriaDataset(img_dir='/data/yjt/mydatasets/images/val/',
                                                            lab_dir='/data/yjt/mydatasets/labels/val/',
                                                            max_lab=90,
                                                            imgsize=train_image_size,
                                                            shuffle=True),
                                                batch_size=train_batch_size,
                                                shuffle=True,
                                                num_workers=10)
    train_loader = DeviceDataLoader(train_loader, device)
    # 检测器选择
    detectorfasterrcnn = chooseDector('fasterrcnn')
    detectorssd = chooseDector('ssd')
    # detectoryolov3 = chooseDector('yolov3')
    detectoryolov5 = chooseDector('yolov5')
    detectoryolov7 = chooseDector('yolov7')
    
    # 生成patch 
    adv_patch_cpu = torch.full((3,150,150), 0.5)
    adv_patch_cpu.requires_grad_(True)
    
    # adv_patch = torch.sigmoid(adv_patch)
    # adv_patch = torch.nn.Parameter(torch.rand(3,10, 10,device = torch.device('cuda')),requires_grad=True)
    # adv_patch_act=torch.sigmoid(adv_patch)
    # adv_patch_act.requires_grad_(True)

    # patch 应用器 增强器 和 TV计算器
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    
    # 定制学习率
    start_lr = 0.01
    iter_max = 800000 #最大训练步数 (留一个没有根据这个退出的bug)学习到步骤了
    power = 0.9
    iter_batch=0  #迭代步数
    iter_collect=10 #集体攻击步数
    iter_save=1000 #每过1000次保存一次patch 
    # lr = start_lr*(1-iter_batch/iter_max)**power 调整策略 

    # 优化器
    optimizer = torch.optim.Adam([adv_patch_cpu], lr=start_lr, betas=(0.5, 0.999), amsgrad=True)

    # 开始训练
    iteration_total = len(train_loader)
    torch.cuda.empty_cache()
    start_epoch=1
    n_epochs=10000
    pr_loss = []
    
    for epoch in range(start_epoch, n_epochs+1):
        for i,(img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch} {pr_loss}',total=iteration_total):
            iter_batch=iter_batch+1
            with autograd.detect_anomaly():
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                adv_patch =adv_patch_cpu.cuda()
                iter_single=0

                # patch应用 mask_patch->adv_batch->同一个batch的不同视角？实际上就是不同视角的
                adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, do_rotate=True, rand_loc=False)[0]
                p_img_batch = patch_applier(img_batch, adv_batch_t)


                loss_det_fasterrcnn=detectorfasterrcnn.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=512)
                loss_det_ssd = detectorssd.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=300)
                # loss_det_yolov3 = detectoryolov3.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=416)
                loss_det_yolov5 = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                loss_det_yolov7 = detectoryolov7.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
    
                loss_det = 0.5*loss_det_fasterrcnn+0.25*loss_det_ssd+0.25*loss_det_yolov5+0.5*loss_det_yolov7
                # loss_det =    0.5*torch.mean(loss_det_yolov5)+0.5*torch.mean(loss_det_yolov7)
                loss = loss_det

                print('fasterrcnn:',loss_det_fasterrcnn.detach().cpu().numpy(),'\t',
                    'ssd:',loss_det_ssd.detach().cpu().numpy(),'\t',
                    'yolov5:',loss_det_yolov5.detach().cpu().numpy(),'\t',
                    'yolov7:',loss_det_yolov7.detach().cpu().numpy(),'\t',
                    'learining rate:',optimizer.param_groups[0]['lr']
                    )
                
                loss.backward()
                optimizer.step()
                optimizer.param_groups[0]['lr']=start_lr*(1-iter_batch/iter_max)**power 
                adv_patch_cpu.data.clamp_(0,1)
                writer.add_scalar('loss',loss, iter_batch)

                # 集成
                print("collect: \n")
                while loss.item()>0.6 and iter_single<20 and iter_batch<iter_collect: 
                    iter_batch=iter_batch+1
                    iter_single=iter_single+1 #while 控制

                    adv_patch =adv_patch_cpu.cuda()
                    adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, do_rotate=False, rand_loc=False)[0]
                    p_img_batch = patch_applier(img_batch, adv_batch_t)
                    
                    loss_det_fasterrcnn=detectorfasterrcnn.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=512)
                    loss_det_ssd = detectorssd.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=300)
                    loss_det_yolov5 = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                    loss_det_yolov7 = detectoryolov7.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)

                    loss_det = torch.mean(0.5*loss_det_fasterrcnn+0.25*loss_det_ssd+0.25*loss_det_yolov5+0.5*loss_det_yolov7)
                    loss =loss_det
                    loss.backward()
                    optimizer.step()
                    optimizer.param_groups[0]['lr']=start_lr*(1-iter_batch/iter_max)**power

                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)
                    print('fasterrcnn:',loss_det_fasterrcnn.detach().cpu().numpy(),'\t',
                    'ssd:',loss_det_ssd.detach().cpu().numpy(),'\t',
                    'yolov5:',loss_det_yolov5.detach().cpu().numpy(),'\t',
                    'yolov7:',loss_det_yolov7.detach().cpu().numpy(),'\t',
                    'learining rate:',optimizer.param_groups[0]['lr']
                    )

                # 轮流攻击  0.25为检测置信度
                print("yolov7: \n")
                iter_single=0
                while loss_det_yolov7.item()>0.25 and iter_single<20 and iter_batch>iter_collect:
                    iter_batch=iter_batch+1
                    iter_single=iter_single+1 #while 控制

                    adv_patch =adv_patch_cpu.cuda()
                    adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, do_rotate=False, rand_loc=False)[0]
                    p_img_batch = patch_applier(img_batch, adv_batch_t)
                    loss_det_fasterrcnn=detectorfasterrcnn.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=512)
                    loss_det_ssd = detectorssd.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=300)
                    loss_det_yolov5 = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                    loss_det_yolov7 = detectoryolov7.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                    # loss_det = torch.mean(0.5*loss_det_fasterrcnn+0.25*loss_det_ssd+0.25*loss_det_yolov5+0.5*loss_det_yolov7)
                    loss =loss_det_yolov7
                    loss.backward()
                    optimizer.step()
                    optimizer.param_groups[0]['lr']=start_lr*(1-iter_batch/iter_max)**power

                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)
                    print('fasterrcnn:',loss_det_fasterrcnn.detach().cpu().numpy(),'\t',
                    'ssd:',loss_det_ssd.detach().cpu().numpy(),'\t',
                    'yolov5:',loss_det_yolov5.detach().cpu().numpy(),'\t',
                    'yolov7:',loss_det_yolov7.detach().cpu().numpy(),'\t',
                    'learining rate:',optimizer.param_groups[0]['lr']
                    )

                print("yolov5: \n")
                iter_single=0
                while loss_det_yolov5.item()>0.25 and iter_single<20 and iter_batch>iter_collect:
                    iter_batch=iter_batch+1
                    iter_single=iter_single+1 #while 控制

                    adv_patch =adv_patch_cpu.cuda()
                    adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, do_rotate=False, rand_loc=False)[0]
                    p_img_batch = patch_applier(img_batch, adv_batch_t)
                    
                    loss_det_fasterrcnn=detectorfasterrcnn.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=512)
                    loss_det_ssd = detectorssd.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=300)
                    loss_det_yolov5 = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                    loss_det_yolov7 = detectoryolov7.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)

                    # loss_det = torch.mean(0.5*loss_det_fasterrcnn+0.25*loss_det_ssd+0.25*loss_det_yolov5+0.5*loss_det_yolov7)
                    loss =loss_det_yolov5
                    loss.backward()
                    optimizer.step()
                    optimizer.param_groups[0]['lr']=start_lr*(1-iter_batch/iter_max)**power

                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)
                    print('fasterrcnn:',loss_det_fasterrcnn.detach().cpu().numpy(),'\t',
                    'ssd:',loss_det_ssd.detach().cpu().numpy(),'\t',
                    'yolov5:',loss_det_yolov5.detach().cpu().numpy(),'\t',
                    'yolov7:',loss_det_yolov7.detach().cpu().numpy(),'\t',
                    'learining rate:',optimizer.param_groups[0]['lr']
                    )
                
                print("ssd: \n")
                iter_single=0
                while loss_det_ssd.item()>0.25 and iter_single<20 and iter_batch>iter_collect:
                    iter_batch=iter_batch+1
                    iter_single=iter_single+1 #while 控制

                    adv_patch =adv_patch_cpu.cuda()
                    adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, do_rotate=False, rand_loc=False)[0]
                    p_img_batch = patch_applier(img_batch, adv_batch_t)
                    
                    loss_det_fasterrcnn=detectorfasterrcnn.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=512)
                    loss_det_ssd = detectorssd.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=300)
                    loss_det_yolov5 = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                    loss_det_yolov7 = detectoryolov7.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)

                    # loss_det = torch.mean(0.5*loss_det_fasterrcnn+0.25*loss_det_ssd+0.25*loss_det_yolov5+0.5*loss_det_yolov7)
                    loss =loss_det_ssd
                    loss.backward()
                    optimizer.step()
                    optimizer.param_groups[0]['lr']=start_lr*(1-iter_batch/iter_max)**power

                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)
                    print('fasterrcnn:',loss_det_fasterrcnn.detach().cpu().numpy(),'\t',
                    'ssd:',loss_det_ssd.detach().cpu().numpy(),'\t',
                    'yolov5:',loss_det_yolov5.detach().cpu().numpy(),'\t',
                    'yolov7:',loss_det_yolov7.detach().cpu().numpy(),'\t',
                    'learining rate:',optimizer.param_groups[0]['lr']
                    )
                
                print("fasterrcnn: \n")
                iter_single=0
                while loss_det_fasterrcnn.item()>0.25 and iter_single<20 and iter_batch>iter_collect:
                    iter_batch=iter_batch+1
                    iter_single=iter_single+1 #while 控制

                    adv_patch =adv_patch_cpu.cuda()
                    adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, do_rotate=False, rand_loc=False)[0]
                    p_img_batch = patch_applier(img_batch, adv_batch_t)
                    
                    loss_det_fasterrcnn=detectorfasterrcnn.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=512)
                    loss_det_ssd = detectorssd.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=300)
                    loss_det_yolov5 = detectoryolov5.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)
                    loss_det_yolov7 = detectoryolov7.attack(input_imgs=p_img_batch, attack_id=[4,7,80], total_cls=85,object_thres=0.1,clear_imgs=img_batch,img_size=640)

                    # loss_det = torch.mean(0.5*loss_det_fasterrcnn+0.25*loss_det_ssd+0.25*loss_det_yolov5+0.5*loss_det_yolov7)
                    loss =loss_det_fasterrcnn
                    loss.backward()
                    optimizer.step()
                    optimizer.param_groups[0]['lr']=start_lr*(1-iter_batch/iter_max)**power

                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)
                    print('fasterrcnn:',loss_det_fasterrcnn.detach().cpu().numpy(),'\t',
                    'ssd:',loss_det_ssd.detach().cpu().numpy(),'\t',
                    'yolov5:',loss_det_yolov5.detach().cpu().numpy(),'\t',
                    'yolov7:',loss_det_yolov7.detach().cpu().numpy(),'\t',
                    'learining rate:',optimizer.param_groups[0]['lr']
                    )
                
                if iter_batch%iter_save==0:
                    print('save generated patch\n')
                    patch_save = np.clip(np.transpose(adv_patch_cpu.detach().numpy(), (1, 2, 0)), 0, 1)
                    patch_save = Image.fromarray(np.uint8(255*patch_save))
                    patch_save.save("training_patches/" + str(iter_batch) + " patch.png")
                
                # 保存每个batch的图像
                p_img_batch_np = p_img_batch.detach().cpu().numpy()
                p_img_batch_np=np.transpose(p_img_batch_np,(0,2,3,1))
                p_img_batch_np=np.uint8(p_img_batch_np[0,:,:,:]*255)
                imwrite('./res_imgs/'+str(i)+'.png',p_img_batch_np)
                del adv_batch_t,loss_det, p_img_batch, loss
                torch.cuda.empty_cache()

                if iter_batch>iter_max:
                    print('training finished! \n')
                    break
        optimizer.zero_grad()


def chooseDector(str):
    """
    输入string 返回检测器,
    每个检测器存在一个attack攻击方法 返回 [Batch_size,attack_id]
    """
    if  str=='yolov3':
        # cfgfile = '/data/yjt/adversarial_attack/myattack/PyTorchYOLOv3/config/yolov3.cfg'
        # weightfile='../data/myattack_training_models/yolov3_ckpt_50.pth/yolov3.weights'
        # detector=MyDetectorYolov3(cfgfile,weightfile)
        # return detector 
        pass 

    elif str=='yolov5':
        cfgfile = '/data/yjt/adversarial_attack/myattack/PyTorchYOLOv5/models/yolov5s.yaml'
        weightfile='../myattack_training_models/yolov5s-85-dict.pt'
        detector=MyDetectorYoLov5(cfgfile,weightfile)
        return detector
    
    elif str=='yolov7':
        cfgfile = '/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/cfg/training/yolov7.yaml'
        weightfile='../myattack_training_models/yolov7-85-dict.pt'
        detector =MyDetectorYoLov7(cfgfile,weightfile)
        return detector
    
    elif str=='fasterrcnn':
        weightfile = '../myattack_training_models/resNetFpn-model-15.pth'
        detector = MyFastercnn(weightfile)
        return detector
    
    elif str=='ssd':
        weightfile = '../myattack_training_models/ssd300-40.pth'
        detector = MySSD(weightfile)
        return detector

# patch生成
def generate_patch(opt):
    if opt.generator == 'gray':
        adv_patch_cpu = torch.full((3,150,150), 0.5)
        return adv_patch_cpu
    elif opt.generator == 'random':
        adv_patch_cpu = torch.rand((3,150,150))
        return adv_patch_cpu
    elif opt.generator == 'tanh':
        return None 


if __name__ =='__main__':
    opt=parser_opt(known=True)
    run(opt)


