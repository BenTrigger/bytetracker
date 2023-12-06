import numpy as np
import torch

import sys
sys.path.append('yolov5_v7')
from yolov5_v7.models.common import DetectMultiBackend
from yolov5_v7.utils.general import check_img_size
from yolov5_v7.utils.augmentations import letterbox
from yolov5_v7.utils.general import non_max_suppression, scale_boxes, det_bt_format, fit_boxes
from yolov5_v7.utils.torch_utils import select_device
sys.path.append('ByteTracker')
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from deploy.config.config import init_params
from deploy.algos.euclidean_algo import Euclidean

#from tracker_engine.tracker_engine_abc.tracker_engine_abc.tracker_icd_types import BoundingBox
from utils.utils import BoundingBox

class ByteTracker:
    def __init__(self, image_size=None, config_file="tracker_engine/byte_tracker/byte_tracker/deploy/config/bytetracker_params.yaml"):
        self.params = init_params(config_file)
        if image_size is not None:
            self.params.original_imgsz = image_size
        self.model = DetectMultiBackend(self.params.yolo_model, device=select_device(self.params.device), fp16=self.params.half)
        self.bytetracker = BYTETracker(self.params)
        # if we use cropped image or full image
        self.imgsz = self.params.crop_imgsz if self.params.crop_img else self.params.original_imgsz
        print('self.imgsz: %s'%self.imgsz)
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride)

        print("starting yolo warmup...")
        self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else 1, 3, * self.imgsz))  # warmup
        print("done")

    def flip_treatment(self, img, i_j_arr):
        self.params.original_imgsz = img.shape[:2]
        img_after_flip = self.crop_and_fit(i_j_arr, img)

        # Yolo Model
        preds = self.model(img_after_flip, augment=self.params.augment, visualize=False)
        preds = non_max_suppression(preds, conf_thres=self.params.conf_thres, iou_thres=self.params.iou_thres, classes=self.params.classes, agnostic=False, max_det=100)
        online = []
        dets = []
        #print('preds is %s' % preds)
        for i, det in enumerate(preds):
            if det is not None and len(det):
                print('Yolo Detected!')
                # img.shape = original image shape and not cropped image !!!
                #det[:, :4] = scale_boxes(img_after_flip.shape[2:], det[:, :4], img.shape).round()
                det[:, :4] = fit_boxes(self.imgsz, det[:, :4], i_j_arr, img.shape[:2], self.params.original_imgsz).round()
                #xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                #clss = det[:, 5]
                online_targets = self.bytetracker.update(det.detach().cpu().numpy(), self.params.original_imgsz, self.params.original_imgsz)

                online.append(self.online_handle(det, online_targets))
                det_format = det_bt_format(det.detach().cpu().numpy())
                dets.append(det_format)
            else:
                print("No Detections in this image, return None")
                return None
        print('online: %s' % online)
        print('type of online: %s'% type(online))
        print('type of online[0]: %s'% type(online[0]))
        if len(online[0]):
            [x1, y1, w, h] = Euclidean().get_best_pred_by_indication(online, i_j_arr) # sending bytetracker bboxes
        else:
            [x1, y1, w, h] = Euclidean().get_best_pred_by_indication(dets, i_j_arr) # sending ATR bboxes

        #print(img.shape)
        #print('image bbox are: %s, %s, %s ,%s '%(x1, y1, w, h))
        #cv2.rectangle(img, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color=(255, 0, 0))
        #print(inx)
        #cv2.imwrite('/app/byte_tracker_for_einat_0.3/deploy/results/' + str(inx) + '.jpg' , img)
        #cv2.putText(img_after_flip, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9)
        return BoundingBox(x1, y1, w, h, confs)  # BoundingBox is Top Left Widgh Hight FORMAT

    # def bytetracker_handle(self, det, xywhs):
    #     multiple_num = float(self.params.tracker_lowfps)
    #     if self.params.tracker_lowfps > 1:  # multiple_det_for_low_fps
    #         xywhs[:,2] *= multiple_num
    #         xywhs[:,3] *= multiple_num
    #         xyxy_multiple = xywh2xyxy(xywhs)
    #         det[:,0:4] = xyxy_multiple
    #         online_targets = self.bytetracker.update(det.detach().cpu().numpy(), self.imgsz, self.imgsz)
    #         xywhs[:, 2] /= multiple_num
    #         xywhs[:, 3] /= multiple_num
    #         xyxy_multiple = xywh2xyxy(xywhs)
    #         det[:, 0:4] = xyxy_multiple
    #     else:
    #         online_targets = self.bytetracker.update(det.detach().cpu().numpy(), self.imgsz, self.imgsz)
    #
    #    return online_targets

    def online_handle(self, det, online_targets):
        #multiple_num = float(self.params.tracker_lowfps)
        online = []
        for _, t in enumerate(online_targets):
            # if self.params.tracker_lowfps > 1:
            #     x1, y1, x2, y2 = t.tlbr
            #     new_center_x = x1 + (x2-x1) / 2.
            #     new_center_y = y1 + (y2-y1) / 2.
            #     w_tmp = (x2-x1) / multiple_num
            #     h_tmp = (y2-y1) / multiple_num
            #     x1 = new_center_x - w_tmp/2.
            #     y1 = new_center_y - h_tmp/2.
            #     x2 = new_center_x + w_tmp/2.
            #     y2 = new_center_y + h_tmp/2.
            # else:
            x1, y1, x2, y2 = t.tlbr
            tid = t.track_id
            score = t.score
            cls = t.clss
            # CHANGE THE BBOX TO BE YOLO DETECTION and not kalman filter like it works for bytetracker !!! BEN
            for one_det in det:
                one_conf = one_det[4]
                one_class = one_det[5]
                if round(float(one_conf), 2) == round(float(score), 2) and int(one_class) == int(cls):
                    x1, y1, x2, y2 = one_det[0:4].cpu()  # one_det[1] , one_det[2] , one_det[3]
            online.append(torch.tensor([x1, y1, x2, y2, tid, cls, score]))
            # END

        if len(online) > 0:
            online = np.stack(online, axis=0)
        outputs = online
        return outputs

    def image_fit_yolo(self, model, img):
        print('stride is : ' + str(self.model.stride))
        # From DateLoader
        print(str(img.shape))
        im = letterbox(img, self.params.original_imgsz, stride=self.model.stride, auto=self.model.pt)[0]  # padded resize , self.imgsz
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        print(str(im.shape))
        # Start of loop
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3: # for batch
            im = im[None]
        return im

    def crop_and_fit(self, ref_i_j_point, img):
        img = self.image_fit_yolo(self.model, img)
        if self.params.crop_img: # If we want to crop the original image to a smaller size
            width_start = int(ref_i_j_point[0] - self.imgsz[0]/2)
            width_end = int(ref_i_j_point[0] + self.imgsz[0]/2)
            height_start = int(ref_i_j_point[1] - self.imgsz[1]/2)
            height_end = int(ref_i_j_point[1] + self.imgsz[1]/2)
            print('width_start: %s, width_end: %s ,height_start: %s , height_end: %s '%(width_start,width_end,height_start, height_end ))
            #if len(img.shape) == 3:  #
            #    img = img[None]
            img = img[:, :, height_start:height_end, width_start:width_end]
        print(str(img.shape))
        #img = self.image_fit_yolo(self.model, img)
        return img





