import numpy as np
import torch
import argparse

#from config.project_config import init_params
from .yolov5_v7.models.common import DetectMultiBackend
from .yolov5_v7.utils.general import check_img_size, xyxy2xywh, xywh2xyxy
from .yolov5_v7.utils.augmentations import letterbox
from .yolov5_v7.utils.general import non_max_suppression, scale_boxes


# DELETE IT and import from ROS api
class BoundingBox:
    def __init__(self, x=0, y=0, w=0, h=0, conf=0.0):
        self.x_offset = x  # Leftmost pixel of the ROI # (0 if the ROI includes the left edge of the image)
        self.y_offset = y  # Topmost pixel of the ROI # (0 if the ROI includes the top edge of the image)
        self.width = w  # Width of ROI
        self.height = h  # Height of ROI
        self.confidence_level = conf  # confidence of detection 0-1 float

    def set_bbox(self, x, y, w, h, conf=0.0):
        self.x_offset = x  # Leftmost pixel of the ROI # (0 if the ROI includes the left edge of the image)
        self.y_offset = y  # Topmost pixel of the ROI # (0 if the ROI includes the top edge of the image)
        self.width = w  # Width of ROI
        self.height = h  # Height of ROI
        self.confidence_level = conf  # confidence of detection 0-1 float


class ByteTracker:
    def __init__(self, config_file="bytetracker_params.yaml", crop_img=[640, 480]):
        #self.params = init_params(config_file, 1)
        self.params = define_const_args()
        self.model = DetectMultiBackend(self.params.yolo_model, device=self.params.device, dnn=self.params.dnn, data=self.params.data, fp16=self.params.half)
        imgsz = check_img_size(self.params.imgsz, s=self.model.stride)
        print("starting yolo warmup...")
        self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else 1, 3, *imgsz))  # warmup
        print("done")

    def flip_treatment(self, img, i_j_arr):
        img_after_flip = self.img_reshape(i_j_arr, img, img.shape)

        # Yolo Model
        preds = self.model(img_after_flip, augment=self.params.augment, visualize=False)
        preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=100)

        for i, det in enumerate(preds):
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img_after_flip.shape[2:], det[:, :4], img.shape).round() #TODO - check if scalse coords instead!
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                Online_targerts = self.bytetracker_handle(det, xywhs)
                multiple_num = float(self.params.tracker_lowfps)
                online = self.online_handle(det, Online_targerts, multiple_num)
        new_bbox = euclidean_algo(preds, i_j_arr, img_after_flip, img.shape, device=self.params.device)
        return BoundingBox(new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], confs)  # BoundingBox is Top Left Widgh Hight FORMAT


    def bytetracker_handle(self, det, xywhs):
        multiple_num = float(self.params.tracker_lowfps)
        if self.params.tracker_lowfps > 1:  # multiple_det_for_low_fps
            xywhs[:,2] *= multiple_num
            xywhs[:,3] *= multiple_num
            xyxy_multiple = xywh2xyxy(xywhs)
            det[:,0:4] = xyxy_multiple
            online_targets = self.bytetracker.update(det.detach().cpu().numpy(), self.params.imgsz, self.params.imgsz)
            xywhs[:, 2] /= multiple_num
            xywhs[:, 3] /= multiple_num
            xyxy_multiple = xywh2xyxy(xywhs)
            det[:, 0:4] = xyxy_multiple
        else:
            online_targets = self.bytetracker.update(det.detach().cpu().numpy(), self.params.imgsz, self.params.imgsz)

        return online_targets

    def online_handle(self, det, online_targets, multiple_num):
        online = []
        for _, t in enumerate(online_targets):
            if self.params.tracker_lowfps > 1:
                x1, y1, x2, y2 = t.tlbr
                new_center_x = x1 + (x2-x1) / 2.
                new_center_y = y1 + (y2-y1) / 2.
                w_tmp = (x2-x1) / multiple_num
                h_tmp = (y2-y1) / multiple_num
                x1 = new_center_x - w_tmp/2.
                y1 = new_center_y - h_tmp/2.
                x2 = new_center_x + w_tmp/2.
                y2 = new_center_y + h_tmp/2.
            else:
                x1, y1, x2, y2 =  t.tlbr
            tid = t.track_id
            score = t.score
            cls = t.clss
            # ENTER HERE FIX OF BBOX LIKE YOLO DETECTION and not kalman filter like bytetracker !!! BEN
            for one_det in det:
                one_conf= one_det[4]
                one_clss = one_det[5]
                if round(float(one_conf),2) == round(float(score),2) and int(one_clss) == int(cls):
                    x1, y1, x2, y2 = one_det[0:4].cpu()#, one_det[1] , one_det[2] , one_det[3]
            online.append(torch.tensor([x1, y1, x2, y2, tid, cls, score]))
            # END

        if len(online) > 0:
            online = np.stack(online, axis=0)
        outputs = online
        return outputs

    def image_fit_yolo(self, model, img, imgsz):
        print('stride is : ' +  str(self.model.stride))
        im = letterbox(img, imgsz, stride=self.model.stride,  auto=self.model.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3: #
            im = im[None]
        return im

    def img_reshape(self, ref_i_j_point, img, orig_imgsz):
        config_imgsz = self.params.imgsz
        if self.params.crop_ref: # If we want to crop the original image to a smaller size
            print('crop mode')
            height_start = int(ref_i_j_point[0] - config_imgsz[0]/2)
            height_end = int(ref_i_j_point[0] + config_imgsz[1]/2)
            width_start = int(ref_i_j_point[1] - config_imgsz[0]/2)
            width_end = int(ref_i_j_point[1] + config_imgsz[1]/2)
            img_after_flip = self.image_fit_yolo(self.model, img, orig_imgsz)
            img_after_flip = img_after_flip[:,:, height_start:height_end, width_start:width_end]
        else:  # If we want to reshape the original image to a smaller size
            print('resize mode')
            img_after_flip = self.image_fit_yolo(self.model, img, config_imgsz)
        return img_after_flip

def euclidean_algo(boxes, ref_point, img_after_flip, orig_imgsz, device):
    def xyxy_to_xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = (x[0] + x[2]) / 2  # x center
        y[1] = (x[1] + x[3]) / 2  # y center
        y[2] = x[2] - x[0]  # width
        y[3] = x[3] - x[1]  # height
        return y
    def xy_topleft(box):
        # from xy centered form to xy top left
        box_tl = []
        box_tl.append(int(box[0] -box[2]/2))
        box_tl.append(int(box[1] -box[3]/2))
        box_tl.append(box[2])
        box_tl.append(box[3])
        return box_tl
    def xyxy_to_ij(boxes):
        # takes boxes in xywh format, convert to ij format, together with the eqiuvelant score
        boxes_ij = []
        for box in boxes:
            i = int((box[0] + box[2])/2) # 1 , 3
            j = int((box[1] + box[3])/2) # 2 , 4
            box_ij = [i,j, box[4]] # 5
            print('box is ' + str(box))
            print('ij are' + str(box_ij))
            boxes_ij.append(box_ij)
        return boxes_ij
    def euclidean(box , ref_point):
        # calculate euclidean distance between box in ij format to the referance point
        dist = 0
        for i, _ in enumerate (box):
            dist += (box[i] -ref_point[i])**2
        dist = dist**(0.5)
        print('box is ' + str(box))
        print('ij are' + str(ref_point))
        print('distance is ' + str(dist))
        return dist
    def score_calc(boxes, ij_boxes, ref_point):
        # calculate the score of each bbox considering ref point, then return the box with the maximal score
        final_scores = []
        for box in ij_boxes:
            print('box is ' + str(box))
            final_scores.append(box[2]/euclidean(box[0:2], ref_point))
            print('final score is ' + str((box[2]/euclidean(box[0:2], ref_point))))
        print('final scores are' + str(final_scores))
        arg_max = np.argmax(final_scores)
        bbox = boxes[arg_max]
        return bbox
    def Default_bbox(orig_imgsz, w, h):
        default_box = [] #orig_imgsz - original image's size [h, w, c]
        default_box.append(int(orig_imgsz[1]/2 - w/2))
        default_box.append(int(orig_imgsz[0]/2 - h/2))
        default_box.append(int(orig_imgsz[1]/2 + w/2))
        default_box.append(int(orig_imgsz[0]/2 + h/2))
        return default_box
    def flip_handle(boxes, ref_point, image, orig_imgsz):
        if boxes[0].numel():
            boxes = check_label(boxes, image)
            ij_boxes = xyxy_to_ij(boxes)
            print('ij boxes are' + str(ij_boxes))
            box_chosen = score_calc(boxes ,ij_boxes, ref_point)
        else:
            print('No relevant boxes from ATR, choosing default ref point naive box!!')
            box_chosen = Default_bbox(orig_imgsz,  w=35, h=35)
            box_chosen = np.array(box_chosen)
        return box_chosen
    def check_label(boxes, image):
    # Check boxes are with label 0 - drones only!
        boxes_label = []
        for box in boxes:
            print('box is ' + str(box))
            box = box[0].cpu().detach().numpy()
            print('box after ' + str(box))
            if (box[5] == 0):
                # box[1] = box[1]*image.shape[3] #1
                # box[2] = box[2]*image.shape[2] #0
                # box[3] = box[3]*image.shape[3] #1
                # box[4] = box[4]*image.shape[2] #0
                boxes_label.append(box)
        return boxes_label
    def boxes_from_atr(boxes, ref_point, image, orig_imgsz, device):
        # Inputs : boxes - list of boxes from ATR, ref_point - referance point in ij format
        # output: one box from ATR
        box = flip_handle(boxes, ref_point, image, orig_imgsz)
        box = xyxy_to_xywh(box)
        box = xy_topleft(box)
        box = box[:4]
        return [float(b) for b in box]

    return boxes_from_atr(boxes, ref_point, img_after_flip, orig_imgsz, device)

def define_const_args():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('--tracker-name', type=str, default='ostrack', help='Name of tracking method.')
    parser.add_argument('--tracker-param', type=str, default='vitb_256_mae_ce_32x4_ep300',
                        help='Name of parameter file.')
    parser.add_argument('--videofile', type=str, default='/home/shai/shai/rocx/soy/SteepDive_Head.mp4',
                        help='path to a video file.')
    parser.add_argument('--track-mode', type=str, default='attack', help='attack or hover mode')
    parser.add_argument('--optional-box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save-results', type=int, default=1, help='Save bounding boxes')
    parser.add_argument('--use-half', type=int, default=0, help='Save bounding boxes')
    parser.add_argument('--use-trt', type=int, default=0, help='Save bounding boxes')
    parser.add_argument('--flip-point', type=int, default=2, help='point of flipping')
    parser.add_argument('--init_bbox', type=list, default=[895, 743, 41, 49], help='initial box for OS tracker')
    parser.add_argument('--ref_point', default=0, help='referance point')
    parser.add_argument("--crop-ref", type=int, default=0,
                        help='Whether to crop the image according to referance point or not')

    # ATR
    parser.add_argument('--yolo_model', nargs='+', type=str, default='Ben_tracker_model_3840.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280, 720], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # ByteTracker
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--ByteTrackerFrames", type=int, default=1, help="test mot20.")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min_track_thresh", type=float, default=0.1, help='threshold for minimum assignment')
    parser.add_argument('--use-bytetracker', action='store_true', default=True, help='use byte tracker or deepsort')
    parser.add_argument('--add-yolobboxes', action='store_true', default=False, help='add a bbox for yolo detections')
    parser.add_argument('--tracker-lowfps', type=int, default=1, help='multiple x4 bbox size to tracker')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt