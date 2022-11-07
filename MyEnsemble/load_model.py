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
    # Standard convolution
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
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

# 直接使用

class Model():
    def __init__(self):
        pass

def attempt_load1(weights, device=None, inplace=True, fuse=True):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:            

        if weights == '/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/yolov7-85.pt':
            with add_path('/data/yjt/adversarial_attack/myattack'):
                with add_path('/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/'):
                    ckpt = torch.load(w, map_location='cpu')  # load
                    # print(sys.path)
        # print("\n")
        # print('end',sys.path)
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

def attempt_load2(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    # from  PyTorchYOLOv5.models.yolo import Detect, Model
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:            
        # print("\n")
        # print('begin',sys.path)
        if weights == '/data/yjt/adversarial_attack/myattack/PyTorchYOLOv5/yolov5s-85.pt':
            with add_path('/data/yjt/adversarial_attack/myattack'):
                with add_path('/data/yjt/adversarial_attack/myattack/PyTorchYOLOv5/'):
                    print(sys.path)
                    ckpt = torch.load(w, map_location='cpu')  # load
        # print("\n")
        # print('end',sys.path)
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode
    # 对于yolov5-5还是这样,对于60.就不是了 下面是关于版本兼容的一些设置... 

    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility


    # Return model
    if len(model) == 1:
        return model[-1]
    
    # 集成
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model


def main():
    with add_path('/data/yjt/adversarial_attack/myattack'):
        # import yaml
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # cfgfile2 = '/data/yjt/adversarial_attack/myattack/PyTorchYOLOv5/models/yolov5s.yaml'
        # hyp = yaml.safe_load('/data/yjt/adversarial_attack/myattack/PyTorchYOLOv5/data/hyps/hyp.scratch-low.yaml')
        # from PyTorchYOLOv5.models.yolo import Model
        # model = Model(cfgfile2 , ch=3, nc=85, anchors=None).to(device)
        # weightfile='/data/yjt/adversarial_attack/myattack/PyTorchYOLOv5/yolov5ss-85.pt'
        # ckpt=torch.load(weightfile,map_location='cpu')
        # model.load_state_dict(ckpt)
        # ok 加载成功了 

        # yolov 7 也能够成功加载，说明最好还是使用state_dict 
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # weightfile='/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/yolov7-85.pt'
        # model=attempt_load1(weightfile)  # load FP32 model
        # torch.save(model.state_dict(), './yolov7-85.pt')
        # print("success v7")
        # 换用第二种方式
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import yaml
        cfgfile2 = '/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/cfg/training/yolov7.yaml'
        hyp = yaml.safe_load('/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/data/hyp.scratch.p5.yaml')
        # anchors来自训练 如果需要可以写在超参数中 或者通过 cfgfile里面获取
        from PyTorchYOLOv7.models.yolo import Model
        model = Model(cfgfile2 , ch=3, nc=85, anchors=None).to(device)
        weightfile='/data/yjt/adversarial_attack/myattack/PyTorchYOLOv7/yolov7-85.pt'
        ckpt=torch.load(weightfile,map_location='cpu')
        model.load_state_dict(ckpt)
        print("sucess v7")


        # model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        # state_dict = ckpt['model'].float().state_dict()  # to FP32
        # state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        # model.load_state_dict(state_dict, strict=False)  # load






        # anchors 从哪里得到的 anchors从 cfg中得到
        # model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)





        # weightfile2='/data/yjt/adversarial_attack/myattack/PyTorchYOLOv5/yolov5s-85.pt'
        # detector2=attempt_load2(weightfile2)
        # # 保存
        # torch.save(detector2.state_dict(), './yolov5ss-85.pt')
        # 加载yolov5
        # cfgfile2 = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/data/coco128.yaml'

# 里面的modules common 冲突怎么解决 临时import？然后取消？

if __name__ =='__main__':
    main()
