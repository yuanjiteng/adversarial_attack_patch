import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs
import cv2

def create_model(num_classes):

    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=86)

    # load train weights
    weights_path = "/data/yjt/adversarial_attack/myattack_training_models/resNetFpn-model-15.pth"   #修改权重路径
    voc_xml_path = '/data/yjt/mydatasets/images/val/'  # 图片路径Airplane  ArmoredVehicle
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])
    model.to(device)
    model.eval()
    
    # read class_indict
    no_target = 'faster_no_target/'
    label_txt = 'faster_pre_label_txt/'
    pre_out  = 'faster_out/'

    mylist = os.listdir(voc_xml_path)
    len_ = len(mylist)

    for i in range(0, len_):
        flag = i
        img_path = voc_xml_path + mylist[flag]
    # load image
        original_img = Image.open(img_path)

        label_json_path = './pascal_voc_classes.json'
        assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
        with open(label_json_path, 'r') as f:
            class_dict = json.load(f)

        category_index = {str(v): str(k) for k, v in class_dict.items()}


        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)
            img_o = cv2.imread(img_path)
            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                if os.path.exists(voc_xml_path[:-7] + no_target) == False:
                    os.makedirs(voc_xml_path[:-7] + no_target)
                out_img = voc_xml_path[:-7] + no_target + mylist[flag]
                cv2.imwrite(out_img,img_o)
            else:
                if os.path.exists(voc_xml_path[:-7] + label_txt) == False:
                    os.makedirs(voc_xml_path[:-7] + label_txt)
                output_path = voc_xml_path[:-7] + label_txt + mylist[i][:-4] + ".txt"
                # sign_ = str(classes)
                with open(output_path, 'w', encoding='utf-8') as f:
                    for i in range(0, predict_boxes.shape[0]):
                        f.write(
                            str(predict_classes[i]) + str(f' {predict_scores[i]} {predict_boxes[i][0]} {predict_boxes[i][1]} {predict_boxes[i][2]} {predict_boxes[i][3]}\n'))

            plot_img = draw_objs(original_img,
                                 predict_boxes,
                                 predict_classes,
                                 predict_scores,
                                 category_index=category_index,
                                 box_thresh=0.3,
                                 line_thickness=20,
                                 font='arial.ttf',
                                 font_size=40)
            if os.path.exists( voc_xml_path[:-7]+pre_out)==False:
                os.makedirs( voc_xml_path[:-7]+pre_out)
            out_img =  voc_xml_path[:-7]+pre_out+mylist[flag]
            plot_img.save(out_img)
            print(i)


if __name__ == '__main__':
    main()
