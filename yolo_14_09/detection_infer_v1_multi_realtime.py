import time
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadImageObjects
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import nvidia_smi  # pip install nvidia-ml-py3
import time


def available_memory():
    num_slices = 15
    im_size = 6220800
    test_size = 18662400
    # img.nelement()

    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        #print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
    nvidia_smi.nvmlShutdown()
    num_imgs = int(info.free/im_size)
    if num_imgs-num_slices > 1:
        return [num_slices]
    else:
        num_batches = int(num_slices/num_imgs)
        last_batch = num_slices/num_imgs - a
        batches = []
        for i in range(num_batches):
            batches.append(num_imgs)
        batches.append(int(num_imgs*last_batch))
        return batches


class YoloDetection:
    def __init__(self, weights='', imgsz=1600 , cuda='', conf_thres=0.25):
        self.weights = weights  # 'model.pt path(s)'
        self.source = ''  # input data src
        self.imgsz = imgsz  # inference size (pixels)
        self.conf_thres = conf_thres  # object confidence threshold
        self.iou_thres = 0.45  # IOU threshold for NMS
        self.project = 'runs/detect'  # save results to project/name
        self.name = 'exp'  # save results to project/name
        self.exist_ok = True  # existing project/name ok, do not increment
        self.device = select_device(cuda)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        check_requirements(exclude=('pycocotools', 'thop'))

        self.dataset = ''
        # Directories
        # save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))  # increment run

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once

    def load_dataset(self, input_data, img_obj=False):
        self.img_obj = img_obj
        if img_obj:
            self.dataset = LoadImageObjects(input_data, img_size=self.imgsz, stride=self.stride)
            self.webcam = False
        else:
            self.source = input_data
            self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
                ('rtsp://', 'rtmp://', 'http://'))
            if self.webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)

            else:
                self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)

    def run_inference(self, starting_pixels, imgs, classes=None, size=1600):
        # Run inference
        self.img_obj = True
        t0 = time.time()
        self.webcam = False
        predictions = []
        im0ss = []

        # imgs = False

        # t_slice_to_batch = time.time()
        # for _, im, im0s, _ in self.dataset:
        #     im0ss.append(im0s)
        #     if type(imgs)==bool:
        #         imgs = np.expand_dims(im, axis=0)
        #     else:
        #         im = np.expand_dims(im, axis=0)
        #         imgs = np.concatenate((imgs, im), axis=0)
        # print(f'time for appending slices = {time.time() - t_slice_to_batch} for {imgs.shape[-1]} slices')

        path = ''
        vid_cap = None
        batches = available_memory()
        counter = 0
        for batch in batches:
            gimgs = torch.from_numpy(imgs[counter:counter+batch, :,:,:]).to(self.device)
            gimgs = gimgs.half() if self.half else gimgs.float()  # uint8 to fp16/32
            gimgs /= 255.0  # 0 - 255 to 0.0 - 1.0
            if gimgs.ndimension() == 3:
                gimgs = gimgs.unsqueeze(0)
            preds = self.model(gimgs, augment=False)[0]
            for n in range(preds.shape[0]):
                pred = torch.unsqueeze(preds[n], 0)
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=classes, agnostic=False)
                img = gimgs[counter+n]
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if self.webcam:  # batch_size >= 1
                        p, s, im0, frame = path, '%g: ' % i, imgs[counter+n, :,:,:][i].copy(), self.dataset.count
                    elif self.img_obj:
                        try:
                            p, s, im0, frame = f'{starting_pixels[counter+n][0]}_{starting_pixels[counter+n][1]}_' \
                                               f'{starting_pixels[counter+n][0] + size}_{starting_pixels[counter+n][0] + size}', \
                                               '', imgs[counter+n,:,:,:], getattr(self.dataset, 'frame', 0)
                        except:
                            print(counter+n, imgs.shape[0])
                    else:
                        p, s, im0, frame = path, '', imgs[counter+n, :,:,:], getattr(self.dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    # save_path = str(save_dir / p.name)  # img.jpg
                    # txt_path = str(save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
                    # s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(img.shape[1:])[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size

                        # det[:, :4] = scale_coords(img.shape[1:], det[:, :4], im0.shape).round()
                        det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img.T.shape).round()

                        # # Print results
                        # for c in det[:, -1].unique():
                        #     n = (det[:, -1] == c).sum()  # detections per class
                        #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            w = 0
                            h = 0
                            xmin, ymin, xmax, ymax = torch.tensor(xyxy).cpu()
                            # line = (cls, *xywh, conf)  # if self.save_conf else (cls, *xywh)  # label format
                            line = ({'file': p.name, 'bbox': [[float(xmin + starting_pixels[counter+n][1]),
                                                               float(ymin + starting_pixels[counter+n][0]),
                                                               float(xmax + starting_pixels[counter+n][1]),
                                                               float(ymax + starting_pixels[counter+n][0])],
                                                              [float(w + starting_pixels[counter+n][1]),
                                                               float(h + starting_pixels[counter+n][0])]],
                                     'score': float(conf),
                                     'category': float(cls)})  # if self.save_conf else (cls, *xywh)  # label format
                            if conf > self.conf_thres:
                                predictions.append(line)
            counter += batch

        return predictions


if __name__ == '__main__':
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/MyHomeDir/datasets/tagging_from_noa/16_12_39/vid/images')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--overlap', type=float, default=0.1)
    parser.add_argument('--weights', type=str, default='/MyHomeDir')
    opt = parser.parse_args()

    yv5 = YoloDetection(weights=opt.weights,imgsz=opt.imgsz, cuda='', )
    print(f'Done. ({time.time() - t:.3f}s)')
    t = time.time()

    yv5.load_dataset(opt.data)
    a = yv5.run_inference()
    print(f'Done. ({time.time() - t:.3f}s)')
