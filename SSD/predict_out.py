import os
import json
import time
import cv2

import torch
from PIL import Image
import matplotlib.pyplot as plt

import transforms
from src import SSD300, Backbone
from draw_box_utils import draw_objs


def create_model(num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # create model
    # 目标检测数 + 背景

    model = create_model(num_classes=86)

    # load train weights
    train_weights = "./save_weights/ssd300-49.pth"
    model.load_state_dict(torch.load(train_weights, map_location='cpu')['model'])
    model.to(device)
    model.eval()


    # read class_indict
    json_path = "./pascal_voc_classes.json"
    assert os.path.exists(json_path), "file '{}' dose not exist.".format(json_path)
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {str(v): str(k) for k, v in class_dict.items()}


    no_target = 'ssd49_no_target_/'
    label_txt = 'ssd49_pre_label_txt/'
    pre_out  = 'ssd49_out/'


    #Tank
    voc_xml_path = 'F:\\Public\\TankAeroplaneAmoredVehicle\\test\\Tank\\images/'   #Airplane  ArmoredVehicle
    voc_xml_path = 'F:/XWF/project/from_yuan/images/'
    mylist = os.listdir(voc_xml_path)
    len_ = len(mylist)

    for i in range(0, len_):
    # load image
        flag = i
        img_path = voc_xml_path + mylist[flag]
        original_img = Image.open(img_path)

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.Resize(),
                                             transforms.ToTensor(),
                                             transforms.Normalization()])
        img, _ = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        with torch.no_grad():
            # initial model
            img_o = cv2.imread(img_path)

            time_start = time_synchronized()
            predictions = model(img.to(device))[0]  # bboxes_out, labels_out, scores_out
            time_end = time_synchronized()
            print("inference+NMS time: {}".format(time_end - time_start))
    #bboxes_out, labels_out, scores_out
            predict_boxes = predictions[0].to("cpu").numpy()
            predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
            predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
            predict_classes = predictions[1].to("cpu").numpy()
            predict_scores = predictions[2].to("cpu").numpy()

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
                                 box_thresh=0.5,
                                 line_thickness=10,
                                 font='arial.ttf',
                                 font_size=20)
            if os.path.exists( voc_xml_path[:-7]+pre_out)==False:
                os.makedirs( voc_xml_path[:-7]+pre_out)
            out_img =  voc_xml_path[:-7]+pre_out+mylist[flag]
            plot_img.save(out_img)
            print(i)


if __name__ == "__main__":
    main()
