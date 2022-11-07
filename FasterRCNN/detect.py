import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import json
import time
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

pwd = Path(os.getcwd()).as_posix()
father_path=os.path.dirname(pwd)
sys.path.remove(os.getcwd())
sys.path.insert(0,father_path)
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from FasterRCNN.draw_box_utils import draw_objs
from FasterRCNN.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from FasterRCNN.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from FasterRCNN.utils.torch_utils import select_device, smart_inference_mode
from FasterRCNN.network_files import FasterRCNN
from FasterRCNN.backbone import resnet50_fpn_backbone

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
def create_model(num_classes):

    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model

@smart_inference_mode()
def run(
        weights=ROOT / 'model_10.pth',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        nosave=False,  # do not save images/videos
        visualize=False,  # visualize features
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        vid_stride=1,  # video frame-rate stride
        font_size=0
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = create_model(num_classes=86)
    model.load_state_dict(torch.load(weights, map_location='cpu')["model"])
    model.eval()
    model.to(device)





    dataset = LoadImages(source)

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}
    for path, im, im0s, vid_cap, s in dataset:


        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

            init_img = torch.zeros((1, 3, im.shape[2],im.shape[2]), device=device)
            model(init_img)
            # 查看大小
            t_start = time_synchronized()
            predictions = model(im.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()






        seen += 1
        if webcam:  # batch_size >= 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '
        else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)
        labels_path = str(save_dir / 'labels' )
        if os.path.exists(labels_path) == False:
            os.makedirs(labels_path)
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
            print(p.name)
        else:

            with open(f'{txt_path}.txt', 'w', encoding='utf-8') as f:
                for i in range(0, predict_boxes.shape[0]):
                    f.write(
                        str(predict_classes[i]) + str(
                            f' {predict_scores[i]} {predict_boxes[i][0]} {predict_boxes[i][1]} {predict_boxes[i][2]} {predict_boxes[i][3]}\n'))
        plot_img = draw_objs(im0s,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=conf_thres,
                             line_thickness=line_thickness,
                             font='arial.ttf',
                             font_size=font_size)
        plot_img.save(save_path)
    print("down")
    print("save at:",save_dir)






def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'model_10.pth', help='model path(s)')
    parser.add_argument('--source', type=str, default='F:\\Public\\TankAeroplaneAmoredVehicle\\test\\Tank\\images/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=20, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--font-size', default=30, type=int, help='font_size')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
