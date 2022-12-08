import argparse
import os
import platform
import sys
import numpy as np
from pathlib import Path
import torch
import json
import time
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
        font_size=0,
        attack_ids=[4, 7, 80],
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
    category_index_list = list(category_index.values())
    # 修改
    failed_predict = 0
    successed_predict = 0
    #tqdm(enumerate(train_loader), desc=f'Running epoch {epoch} {pr_loss}',total=iteration_total)
    for path, im, im0s, vid_cap, s in dataset:
    # for i ,(path, im, im0s, vid_cap, s) in tqdm(enumerate(dataset), total=len(dataset)):


        # Inference

        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

        init_img = torch.zeros((1, 3, im.shape[2],im.shape[2]), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(im.to(device))[0]#outputs = self.model([input_img])[0]
        t_end = time_synchronized()
        # print(s,"inference+NMS time: {}".format(t_end - t_start))
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        for c in np.unique(predict_classes):
            n = (predict_classes == c).sum()  # detections per class
            s += f"{n} {category_index_list[int(c - 1)]}{'s' * (n > 1)}, "  # add to string
        print(s, "inference+NMS time: {}".format(t_end - t_start))






        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path





        keep = np.in1d(predict_classes-1,attack_ids)

        plot_img = draw_objs(im0s,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=conf_thres,
                             line_thickness=line_thickness,
                             font='arial.ttf',
                             font_size=font_size)

        if np.sum(keep)>0:
            successed_predict = successed_predict + 1
            img_save_path = str(save_dir / 'successed_predict')

            if os.path.exists(img_save_path) == False:
                os.makedirs(img_save_path)
            successed_predict_path = img_save_path + '/' + p.name
            plot_img.save(successed_predict_path)
        else:
            failed_predict = failed_predict + 1
            img_save_path = str(save_dir / 'failed_predict')
            if os.path.exists(img_save_path) == False:
                os.makedirs(img_save_path)
            failed_predict_path = img_save_path + '/' + p.name
            plot_img.save(failed_predict_path)

    return successed_predict, failed_predict






def parse_opt():#F:\XWF\project\version5\adversarial_attack_patch\res_imgs
    parser = argparse.ArgumentParser()#F:\XWF\project\version4\adversarial_attack_patch\patch_applier\exp6\images_patch
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'resNetFpn-model-9.pth', help='model path(s)')
    parser.add_argument('--source', type=str, default=r'F:\Public\TankAeroplaneAmoredVehicle\test\imagesEnhanced', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=12, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--font-size', default=20, type=int, help='font_size')
    parser.add_argument('--attack_ids', type=list, default=[4, 7, 80], help='攻击的类别')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt



def parse_opt1():#F:\XWF\project\version5\adversarial_attack_patch\res_imgs
    parser = argparse.ArgumentParser()#F:\XWF\project\version4\adversarial_attack_patch\patch_applier\exp6\images_patch
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'resNetFpn-model-9.pth', help='model path(s)')
    parser.add_argument('--source', type=str, default=r'F:\XWF\project\version4\adversarial_attack_patch\patch_applier2\exp4\images_patch', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=12, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--font-size', default=20, type=int, help='font_size')
    parser.add_argument('--attack_ids', type=list, default=[4, 7, 80], help='攻击的类别')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    return run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    successed_predict, failed_predict=main(opt)
    opt1= parse_opt1()
    successed_predict1, failed_predict1=main(opt1)

    print("干净图像识别成功数目",successed_predict,"干净识别失败数目",failed_predict)
    print("对抗样本识别成功数目",successed_predict1,"对抗样本识别失败数目",failed_predict1)
    print("攻击成功率:",(successed_predict-successed_predict1)/successed_predict)
