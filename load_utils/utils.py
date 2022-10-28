import glob
from pathlib import Path
import re
import sys
import os
import time
import math
from math import pi
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

import struct # get_image_size
import imghdr # get_image_size

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))

def deg_to_rad(deg):
    return deg * pi / 180.0

def rad_to_deg(rad):
    return deg * 180.0 / pi

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    #anchor_step = len(anchors)/num_anchors
    anchor_step = len(anchors)//num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    all_boxes = []
    # print("output size : "+str(output.size()))  # torch.Size([1, 425, 13, 13])
    output = output.view(batch*num_anchors, 5+num_classes, h*w)
    # print("output size : "+str(output.size()))  # torch.Size([5, 85, 169])
    output = output.transpose(0,1).contiguous()
    # print("output size : "+str(output.size()))  # torch.Size([85, 5, 169])
    output = output.view(5+num_classes, batch*num_anchors*h*w)
    # print("output size : "+str(output.size()))  # torch.Size([85, 845])
    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data
    # print(cls_confs.size())
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()
    
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
    
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(round((box[0] - box[2]/2.0) * width))
        y1 = int(round((box[1] - box[3]/2.0) * height))
        x2 = int(round((box[0] + box[2]/2.0) * width))
        y2 = int(round((box[1] + box[3]/2.0) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img

def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('[%i]%s: %f' % (cls_id, class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline = rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img

def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size//5, 5) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
       # remove truths of which the width is smaller then the min_box_scale
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
        t1 = time.time()
        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)
        t2 = time.time()
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
        t1 = time.time()
        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)
        t2 = time.time()
    elif torch.is_tensor(img):
        t1 = time.time()
        t2 = time.time()
    else:
        print("unknown image type at [do_detect]. EXIT")
        exit(-1)

    output = model.forward(img) #Simen: dit doet een forward, vervangen voor duidelijkheid
    # print("output size : "+str(output.size()))  # torch.Size([1, 425, 13, 13])
    #output = output.data
    #for j in range(100):
    #    sys.stdout.write('%f ' % (output.storage()[j]))
    #print('')
    t3 = time.time()

    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors)
    #for j in range(len(boxes)):
        #print(boxes[j])
    t4 = time.time()

    boxes = [nms(box, nms_thresh) for box in boxes]
    t5 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print('             nms : %f' % (t5 - t4))
        print('           total : %f' % (t5 - t0))
        print('-----------------------------------')
    return boxes

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets
      
def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close( )
    return count

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24: 
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2 
                ftype = 0 
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2 
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

# def SwedishFlag_generator(patch_input, c, h, w):
#     #
#     ul_hs = 0
#     ul_he = int(h*(2/5))
#     ul_hd = ul_he - ul_hs
#     ul_ws = 0
#     ul_we = int(w*(3/9))
#     ul_wd = ul_we - ul_ws
#     #
#     ur_hs = 0
#     ur_he = int(h*(2/5))
#     ur_hd = ur_he - ur_hs
#     ur_ws = int(w*(4/9))
#     ur_we = int(w)
#     ur_wd = ur_we - ur_ws
#     #
#     dl_hs = int(h*(3/5))
#     dl_he = int(h)
#     dl_hd = dl_he - dl_hs
#     dl_ws = 0
#     dl_we = int(w*(3/9))
#     dl_wd = dl_we - dl_ws
#     #
#     dr_hs = int(h*(3/5))
#     dr_he = int(h)
#     dr_hd = dr_he - dr_hs
#     dr_ws = int(w*(4/9))
#     dr_we = int(w)
#     dr_wd = dr_we - dr_ws

#     #
#     ulur_mw = int(w*(1/9))
#     dldr_mw = int(w*(1/9))

#     #
#     color_base = [float(255/255),float(164/255),0]

#     # print("patch_input  : "+str(patch_input.size()))
#     patch_input_ul = torch.narrow(torch.narrow(patch_input,-2 ,ul_hs,ul_hd),-1 ,ul_ws,ul_wd).cuda()
#     # print("patch_input_ul : "+str(patch_input_ul.size()))
#     patch_input_ur = torch.narrow(torch.narrow(patch_input,-2 ,ur_hs,ur_hd),-1 ,ur_ws,ur_wd).cuda()
#     # print("patch_input_ur : "+str(patch_input_ur.size()))
#     patch_input_dl = torch.narrow(torch.narrow(patch_input,-2 ,dl_hs,dl_hd),-1 ,dl_ws,dl_wd).cuda()
#     # print("patch_input_dl : "+str(patch_input_dl.size()))
#     patch_input_dr = torch.narrow(torch.narrow(patch_input,-2 ,dr_hs,dr_hd),-1 ,dr_ws,dr_wd).cuda()
#     # print("patch_input_dr : "+str(patch_input_dr.size()))

#     zores_ulurm = torch.zeros((c,ul_hd,ulur_mw)).cuda()
#     zores_ulurm[0,:,:] = zores_ulurm[0,:,:] + color_base[0]
#     zores_ulurm[1,:,:] = zores_ulurm[1,:,:] + color_base[1]
#     zores_ulurm[2,:,:] = zores_ulurm[2,:,:] + color_base[2]
#     # print("zores_ulurm : "+str(zores_ulurm.size()))
#     ulur_ = torch.cat((patch_input_ul, zores_ulurm),dim=2).cuda()
#     # print("ulur_ : "+str(ulur_.size()))
#     ulur_ = torch.cat((ulur_, patch_input_ur),dim=2).cuda()
#     # print("ulur_ : "+str(ulur_.size()))

#     zores_dldrm = torch.zeros((c,dl_hd,dldr_mw)).cuda()
#     zores_dldrm[0,:,:] = zores_dldrm[0,:,:] + color_base[0]
#     zores_dldrm[1,:,:] = zores_dldrm[1,:,:] + color_base[1]
#     zores_dldrm[2,:,:] = zores_dldrm[2,:,:] + color_base[2]
#     # print("zores_dldrm : "+str(zores_dldrm.size()))
#     dldr_ = torch.cat((patch_input_dl, zores_dldrm),dim=2).cuda()
#     # print("dldr_ : "+str(dldr_.size()))
#     dldr_ = torch.cat((dldr_, patch_input_dr),dim=2).cuda()
#     # print("ulur_ : "+str(ulur_.size()))

#     zores_udud = torch.zeros((c,h-ulur_.size()[-2]-dldr_.size()[-2],w)).cuda()
#     zores_udud[0,:,:] = zores_udud[0,:,:] + color_base[0]
#     zores_udud[1,:,:] = zores_udud[1,:,:] + color_base[1]
#     zores_udud[2,:,:] = zores_udud[2,:,:] + color_base[2]
#     # print("zores_udud : "+str(zores_udud.size()))
#     patch_input_ = torch.cat((ulur_, zores_udud),dim=1).cuda()
#     # print("patch_input_ : "+str(patch_input_.size()))
#     patch_input_ = torch.cat((patch_input_, dldr_),dim=1).cuda()
#     # print("patch_input_ : "+str(patch_input_.size()))

#     # print("patch_input_ : "+str(patch_input_.size()))
#     # print("patch_input  : "+str(patch_input.size()))
#     #
#     return patch_input_


def SwedishFlag_generator(patch_input):
    shape = patch_input.size()
    if(len(shape) == 4):
        b, c, h, w = shape
        dim_h = 2
        dim_w = 3
    elif(len(shape) == 3):
        c, h, w = shape
        dim_h = 1
        dim_w = 2
    #
    ul_hs = 0
    ul_he = int(h*(2/5))
    ul_hd = ul_he - ul_hs
    ul_ws = 0
    ul_we = int(w*(3/9))
    ul_wd = ul_we - ul_ws
    #
    ur_hs = 0
    ur_he = int(h*(2/5))
    ur_hd = ur_he - ur_hs
    ur_ws = int(w*(4/9))
    ur_we = int(w)
    ur_wd = ur_we - ur_ws
    #
    dl_hs = int(h*(3/5))
    dl_he = int(h)
    dl_hd = dl_he - dl_hs
    dl_ws = 0
    dl_we = int(w*(3/9))
    dl_wd = dl_we - dl_ws
    #
    dr_hs = int(h*(3/5))
    dr_he = int(h)
    dr_hd = dr_he - dr_hs
    dr_ws = int(w*(4/9))
    dr_we = int(w)
    dr_wd = dr_we - dr_ws

    #
    ulur_mw = int(w*(1/9))
    dldr_mw = int(w*(1/9))

    patch_input_ul = torch.narrow(torch.narrow(patch_input,-2 ,ul_hs,ul_hd),-1 ,ul_ws,ul_wd).cuda()
    patch_input_ul_clone = patch_input_ul.clone().detach().cuda()
    patch_input_ur = torch.narrow(torch.narrow(patch_input,-2 ,ur_hs,ur_hd),-1 ,ur_ws,ur_wd).cuda()
    patch_input_ur_clone = patch_input_ur.clone().detach().cuda()
    patch_input_dl = torch.narrow(torch.narrow(patch_input,-2 ,dl_hs,dl_hd),-1 ,dl_ws,dl_wd).cuda()
    patch_input_dl_clone = patch_input_dl.clone().detach().cuda()
    patch_input_dr = torch.narrow(torch.narrow(patch_input,-2 ,dr_hs,dr_hd),-1 ,dr_ws,dr_wd).cuda()
    patch_input_dr_clone = patch_input_dr.clone().detach().cuda()

    _ulurm = torch.narrow(torch.narrow(patch_input,-2 ,ul_hs,ul_hd),-1 ,ul_we,ulur_mw).cuda()
    _ulurm_clone = _ulurm.clone().detach().cuda()
    
    ulur_ = torch.cat((patch_input_ul, _ulurm_clone),dim=dim_w).cuda()
    ulur_ = torch.cat((ulur_, patch_input_ur),dim=dim_w).cuda()

    ulur_i = torch.cat((patch_input_ul_clone, _ulurm),dim=dim_w).cuda()
    ulur_i = torch.cat((ulur_i, patch_input_ur_clone),dim=dim_w).cuda()

    _dldrm = torch.narrow(torch.narrow(patch_input,-2 ,dl_hs,dl_hd),-1 ,dl_we,dldr_mw).cuda()
    _dldrm_clone = _dldrm.clone().detach().cuda()

    dldr_ = torch.cat((patch_input_dl, _dldrm_clone),dim=dim_w).cuda()
    dldr_ = torch.cat((dldr_, patch_input_dr),dim=dim_w).cuda()

    dldr_i = torch.cat((patch_input_dl_clone, _dldrm),dim=dim_w).cuda()
    dldr_i = torch.cat((dldr_i, patch_input_dr_clone),dim=dim_w).cuda()

    _udud = torch.narrow(torch.narrow(patch_input,-2 ,ul_he,h-ulur_.size()[-2]-dldr_.size()[-2]),-1 ,ul_ws,w).cuda()
    _udud_clone = _udud.clone().detach().cuda()

    patch_input_external = torch.cat((ulur_, _udud_clone),dim=dim_h).cuda()
    patch_input_external = torch.cat((patch_input_external, dldr_),dim=dim_h).cuda()

    patch_input_inner = torch.cat((ulur_i, _udud),dim=dim_h).cuda()
    patch_input_inner = torch.cat((patch_input_inner, dldr_i),dim=dim_h).cuda()

    # print("patch_input_external : "+str(patch_input_external.size()))
    # print("patch_input  : "+str(patch_input.size()))
    #
    return patch_input_external, patch_input_inner