import numpy as np
import torch

from bytetracker.yolov5_v7.models.common import DetectMultiBackend
from bytetracker.yolov5_v7.utils.general import check_img_size, xyxy2xywh, xywh2xyxy
from bytetracker.yolov5_v7.utils.augmentations import letterbox

def ATR_init(weights, device, dnn, data, half, imgsz):

    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
    return model, stride, pt

def bytetracker_handle(opt, det, xywhs, imgsz, bytetracker):
    if opt.use_bytetracker:  ### BYTE Tracker!
        multiple_num = float(opt.tracker_lowfps)
    if opt.tracker_lowfps > 1: #multiple_det_for_low_fps
        xywhs[:, 2] *= multiple_num
        xywhs[:, 3] *= multiple_num
        xyxy_multiple = xywh2xyxy(xywhs)
        det[:, 0:4] = xyxy_multiple
        #print("input for bytetracker det %s" % det[0, 0:4])
        online_targets = bytetracker.update(det.detach().cpu().numpy(), imgsz, imgsz)
        #return it back
        xywhs[:, 2] /= multiple_num
        xywhs[:, 3] /= multiple_num
        xyxy_multiple = xywh2xyxy(xywhs)
        det[:, 0:4] = xyxy_multiple
    else:
        #print("input for bytetracker det %s" % det[0, 0:4])
        online_targets = bytetracker.update(det.detach().cpu().numpy(), imgsz, imgsz)

    return online_targets

def Online_handle(det, opt, online_targets, multiple_num):
    online = []
    for _, t in enumerate(online_targets):
        if opt.tracker_lowfps > 1:
            # tmp = tlwh2xywh(t.tlwh) # it is bigger now ,first we  make xy to center point
            # new_t = [tmp[0], tmp[1], tmp[2] / 4, tmp[3] / 4] # now /4 size of bboxx wh
            # x1, y1, x2, y2 = tlwh_to_xyxy(new_t, imgsz[0], imgsz[1]) # convert it to xyxy
            x1, y1, x2, y2 = t.tlbr
            #print("t.tlbr output: %s" % t.tlbr)
            new_center_x = x1 + (x2-x1) / 2.
            new_center_y = y1 + (y2-y1) / 2.
            w_tmp = (x2-x1) / multiple_num
            h_tmp = (y2-y1) / multiple_num
            x1 = new_center_x - w_tmp/2.
            y1 = new_center_y - h_tmp/2.
            x2 = new_center_x + w_tmp/2.
            y2 = new_center_y + h_tmp/2.
        else:
            #print("t.tlbr output: %s" % t.tlbr)
            x1, y1, x2, y2 =  t.tlbr # tlwh_to_xyxy(t.tlwh, imgsz[0], imgsz[1])
        tid = t.track_id
        score = t.score
        cls = t.clss
        # ENTER HERE FIX OF BBOX LIKE YOLO DETECTION and not kalman filter like bytetracker !!! BEN
        for one_det in det:
            one_conf= one_det[4]
            one_clss = one_det[5]
            if round(float(one_conf),2) == round(float(score),2) and int(one_clss) == int(cls):
                x1, y1, x2, y2 = one_det[0:4].cpu()#, one_det[1] , one_det[2] , one_det[3]
                #print("%s"%(one_det[0:4]))
        online.append(torch.tensor([x1, y1, x2, y2, tid, cls, score]))
        # END
        #online.append(np.array([x1, y1, x2, y2, tid, cls, score]))
    if len(online) > 0:
        online = np.stack(online, axis=0)
    outputs = online
    return outputs

def Image_fit_yolo(model, img, imgsz, stride, pt):
    print('stride is : ' +  str(stride))
    im = letterbox(img, imgsz, stride=stride,  auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None] 
    return im

def img_reshape(crop_ref, ref_point, model, img_ref, imgsz, orig_imgsz, stride, pt):
        if crop_ref: # If we want to crop the original image to a smaller size
            print('crop mode')
            H_start = int(ref_point[0] - imgsz[0]/2)
            H_end = int(ref_point[0] + imgsz[1]/2)
            W_start = int(ref_point[1] - imgsz[0]/2)
            W_end = int(ref_point[1] + imgsz[1]/2)
            #orig_imgsz = check_img_size(orig_imgsz, s=stride)
            img_after_flip = Image_fit_yolo(model, img_ref, orig_imgsz, stride, pt)
            img_after_flip = img_after_flip[:,:, H_start:H_end, W_start:W_end]
            #i_ref, j_ref = new_size[0]/2 ,new_size[1]/2 
        else: # If we want to reshape the original image to a smaller size
            print('resize mode')
            img_after_flip = Image_fit_yolo(model, img_ref, imgsz, stride, pt)
        return img_after_flip

def Euclidean_algo(boxes, ref_point, img_after_flip, orig_imgsz, device):
    

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


    def boxes_from_ATR(boxes, ref_point, image, orig_imgsz, device):
        # Inputs : boxes - list of boxes from ATR, ref_point - referance point in ij format
        # output: one box from ATR
        print('original boxes are ' + str(boxes))
        print('boxes after check are ' + str(boxes))
        Box = flip_handle(boxes, ref_point, image, orig_imgsz)
        print('Box ' + str(Box))
        #Box = xyxy2xywh(Box)
        Box = xyxy_to_xywh(Box)
        print('box xtwh is ' + str(Box))
        Box = xy_topleft(Box)
        Box = Box[:4]
        Box = [float(b) for b in Box]
        #Box = torch.Tensor(Box)
        #Box = Box.tolist()
        print('boxes chosen ' + str(Box))
        return Box
    
    box = boxes_from_ATR(boxes, ref_point, img_after_flip, orig_imgsz, device)
    return box