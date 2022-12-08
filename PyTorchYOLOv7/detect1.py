import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import numpy as np

import os,sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
pwd = Path(os.getcwd()).as_posix()
father_path=os.path.dirname(pwd)
sys.path.remove(os.getcwd())
sys.path.insert(0,father_path)

from PyTorchYOLOv7.models.experimental import attempt_load
from PyTorchYOLOv7.utils.datasets import LoadStreams, LoadImages
from PyTorchYOLOv7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from PyTorchYOLOv7.utils.plots import plot_one_box
from PyTorchYOLOv7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, source_ori,weights, view_img, save_txt, imgsz, trace, = opt.source,opt.source_ori, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    #Statistics
    attack_sum = 0  # 攻击成功数
    failed_attack_sum = 0  # 对抗图像识别成功数目
    all_sum = 1e-10  # 干净图像识别成功的数目
    attack_id = [4, 7, 80]
    save_dir_no_detect = os.path.join(str(save_dir), 'no_detect')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        dataset_ori = LoadImages(source_ori, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    for lhs, rhs in zip(dataset,dataset_ori):
        assert lhs[0].split('/')[-1].split('\\')[-1] == rhs[0].split('/')[-1].split('\\')[-1], f'Ori_Image:{lhs[0]} ,\n Patch_Image:{rhs[0]}'
        path, img, im0s, vid_cap = lhs
        img_ori = rhs[1]

        img_ori = torch.from_numpy(img_ori).to(device)
        img_ori = img_ori.half() if half else img_ori.float()  # uint8 to fp16/32
        img_ori /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_ori.ndimension() == 3:
            img_ori = img_ori.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img_ori.shape[0] or old_img_h != img_ori.shape[2] or old_img_w != img_ori.shape[3]):
            old_img_b = img_ori.shape[0]
            old_img_h = img_ori.shape[2]
            old_img_w = img_ori.shape[3]
            for i in range(3):
                model(img_ori, augment=opt.augment)[0]

        #ori_predict
        pred_ori = model(img_ori, augment=opt.augment)[0]
        pred_ori = non_max_suppression(pred_ori, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        predict_classes_ori = pred_ori[0][:, -1].cpu().numpy()
        predict_scores_ori = pred_ori[0][:, -2].cpu().numpy()
        keep_ori = np.in1d(predict_classes_ori, attack_id)
        if np.sum(keep_ori) <= 0 or np.sum(predict_scores_ori[keep_ori] >= 0.5) <= 0:
            with open(f'{str(save_dir_no_detect)}.txt', 'a', encoding='utf-8') as f:
                f.write(rhs[0])
                f.write('\n')
            continue


        all_sum = all_sum + 1

        # patch_predict
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)



        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # print(pred.shape)
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    predict_classes = pred[0][:, -1].cpu().numpy()
                    predict_scores = pred[0][:, -2].cpu().numpy()
                    keep = np.in1d(predict_classes, attack_id)
                    if np.sum(keep) > 0 and np.sum(predict_scores[keep] >= 0.5) > 0:
                        cv2.imwrite(save_path, im0)
                        failed_attack_sum = failed_attack_sum + 1

                    else:
                        attack_path = str(save_dir / 'attack_success')
                        attack_sum = attack_sum + 1
                        if os.path.exists(attack_path) == False:
                            os.makedirs(attack_path)
                        attack_success_path = str(save_dir / 'attack_success' / p.name)
                        cv2.imwrite(attack_success_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")


    success_rate = attack_sum / all_sum


    print(f'Done. ({time.time() - t0:.3f}s)')
    with open(f'{str(save_dir)}.txt', 'w', encoding='utf-8') as f:
        f.write('The Patch from:')
        f.write(str(source))
        f.write('\n')
        f.write('攻击成功数:')
        f.write(str(attack_sum))
        f.write('\n')
        f.write('对抗图像识别成功数目:')
        f.write(str(failed_attack_sum))
        f.write('\n')
        f.write('success_rate:')
        f.write(str(success_rate))
        f.write('\n')
    print("攻击成功数:", attack_sum)
    print("对抗图像识别成功数目:", failed_attack_sum)
    print("success_rate:", success_rate)

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-85.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='/data/yjt/adversarial_attack/myattack/res_imgs/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source_ori', type=str,
                        default=r'F:\Public\TankAeroplaneAmoredVehicle\test\imagesEnhanced',
                        help='干净图像')
    parser.add_argument('--source', type=str, default=r'F:\XWF\project\version4\adversarial_attack_patch\patch_applier1\exp5\images_patch', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
