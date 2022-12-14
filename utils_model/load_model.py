import math
import numpy as np
import torch
import torch.nn as nn
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

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load1(weights, device=None, inplace=True, fuse=True):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:            
        with add_path('/data1/yjt/adversarial_attack/myattack'):
            with add_path('/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv7/'):
                ckpt = torch.load(w, map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode
    
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    # Module compatibility updates
    # for m in model.modules():
    #     t = type(m)
    #     if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
    #         m.inplace = inplace  # torch 1.7.0 compatibility
    #         if t is Detect and not isinstance(m.anchor_grid, list):
    #             delattr(m, 'anchor_grid')
    #             setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
    #     elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
    #         m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]
    
    # 集成的基本用不上
    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model

# YOLOv5 完全自定义加载方式

def main():
    # YOLOV5 原版加载方式 会增加路径信息（加载存在路径的模型） 
    with add_path('/data1/yjt/adversarial_attack/myattack'):
        from PyTorchYOLOv5.models.common import DetectMultiBackend
        for i in range(30):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weightfile='/data1/yjt/yolov5-master/runs/mytrain/30_models/weights/epoch'+str(i)+'.pt'
            cfgfile = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/models/yolov5s.yaml'
            model = DetectMultiBackend(weightfile, device=device, dnn=False, data=cfgfile, fp16=False)
            print("YOLOv5 load success")
            # print(model.state_dict())
            # 保存方法 注意一定要model.model
            savefile='/data1/yjt/adversarial_attack/myattack_training_models/yolov5s/epoch'+str(i)+'-dict.pt'
            torch.save(model.model.state_dict(),savefile)

    # YOLOv5 第二种加载方式(不存在路径定义)
    # with add_path('/data1/yjt/adversarial_attack/myattack'):
    #     import yaml
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     cfgfile2 = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/models/yolov5s.yaml'
    #     hyp = yaml.safe_load('/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/data/hyps/hyp.scratch-low.yaml')
    #     from PyTorchYOLOv5.models.yolo import Model as YOLOv5Model
    #     model = YOLOv5Model(cfgfile2 , ch=3, nc=85, anchors=None).to(device)
    #     weightfile='/data1/yjt/adversarial_attack/myattack_training_models/yolov5s-85-enhance-dict.pt'
    #     ckpt=torch.load(weightfile,map_location='cpu')
    #     model.load_state_dict(ckpt)
    #     print("success load yolov5")

    # YOLOV7 第一种加载方式 需要增加with 路径
    # with add_path('/data1/yjt/adversarial_attack/myattack'):
    #     with add_path('/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv7/'):
    #         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         import yaml
    #         cfgfile2 = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv7/cfg/training/yolov7.yaml'
    #         hyp = yaml.safe_load('/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv7/data/hyp.scratch.p5.yaml')
    #         # anchors来自训练 如果需要可以写在超参数中 或者通过 cfgfile里面获取
    #         from PyTorchYOLOv7.models.yolo import Model as YOLOv7Model
    #         model = YOLOv7Model(cfgfile2 , ch=3, nc=85, anchors=None).to(device)
    #         weightfile='/data1/yjt/adversarial_attack/myattack_training_models/yolov7-85.pt'
    #         ckpt=torch.load(weightfile,map_location='cpu')
    #         state_dict = ckpt['model'].float().state_dict()
    #         model.load_state_dict(state_dict, strict=False)
    #         print("sucess v7  method1")
    #         torch.save(model.state_dict(), '/data1/yjt/adversarial_attack/myattack_training_models/yolov7-85-dict.pt')

    #YOLOV7 第二种方式，不需要with路径进行加载 
    # with add_path('/data1/yjt/adversarial_attack/myattack'):
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     import yaml
    #     cfgfile2 = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv7/cfg/training/yolov7.yaml'
    #     hyp = yaml.safe_load('/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv7/data/hyp.scratch.p5.yaml')
    #     # anchors来自训练 如果需要可以写在超参数中 或者通过 cfgfile里面获取
    #     from PyTorchYOLOv7.models.yolo import Model as YOLOv7Model
    #     model = YOLOv7Model(cfgfile2 , ch=3, nc=85, anchors=None).to(device)
    #     weightfile='/data1/yjt/adversarial_attack/myattack_training_models/yolov7-85-dict.pt'
    #     ckpt=torch.load(weightfile,map_location='cpu')
    #     model.load_state_dict(ckpt)
    #     print("sucess v7  method2")

if __name__ =='__main__':
    # 不能同时使用两种需要加载路径的方式，否则会报错
    main()
