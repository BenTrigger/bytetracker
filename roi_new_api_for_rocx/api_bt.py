import threading
import argparse
import time
import sys

sys.path.append('bytetracker')
from atr import ATR_init, Euclidean_algo, bytetracker_handle, Online_handle, img_reshape
from bytetracker.yolov5_v7.utils.torch_utils import select_device
from bytetracker.yolov5_v7.utils.general import  non_max_suppression, scale_boxes, xyxy2xywh
from bytetracker.yolov5_v7.models.common import DetectMultiBackend
from bytetracker.yolov5_v7.utils.plots import colors
from bytetracker.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
sys.path.append('os_tracker')
from os_tracker.tracker.lite_tracker import LiteTracker


def tracking_obj(tracker, img, bbox): # maybe get args like this: [tracker, img, bbox]
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        tracker.tracker_mission(img,bbox)
        time.sleep(1)  # we don't need sleep in here, only for example.
    print("Stopping as you wish.")


def init_tracker():
    return LiteTracker()

def init_bytetracker(opt):
    return BYTETracker(opt)


#def run(tracker, bytetracker):
def run(tracker, model):
    flag = True
    args = ("hover",)
    if "attack" == 0:
        img = None # getImg_from_stack
        bbox = None # get BBox from message !
        args = ("attack",)
    tracking_thread = threading.Thread(target=tracker.tracking_obj, args=[tracker,img,bbox])

    while True:  # waiting for messages
        time.sleep(1)
        if flag:
            tracking_thread.start()
            flag = False
        elif "msg to stop showing up" != "":
            time.sleep(5)
            tracking_thread.do_run = False  # Stop message
        print("im still waiting for messages..")


def parse_opt():

    return opt


def main(opt):
    tracker = LiteTracker()
    tracker.warmup()
    device = select_device(0)
    nr_sources = 1
    Bytetrack_frames = opt.ByteTrackerFrames
    outputs = [None] * nr_sources
    imgsz = opt.imgsz
    crop_ref = opt.crop_ref
    device = select_device(opt.device)
    bytetracker = init_bytetracker(opt)
    
    model, stride, pt = ATR_init(opt.yolo_model, device, opt.dnn, opt.half, imgsz)

    #einat data
    #bbox = [895, 743, 41, 49]
    bbox = opt.init_bbox


    tracker.initialize(bbox, 3011)
    tracker.msg = 'tracking'
    counter = 1

#    while not tracker.msg.__eq__('stop'):
    while counter < opt.flip_point:
        t_1 = time.time()
        res = tracker.acquire()  # hover mode
        res_bbox = res['bbox']
        frame_id = res['frame_id']
        print('counter of trakcing: %s' % counter)
        print('bbox: %s , %s , %s, %s' % (res_bbox.x_offset, res_bbox.y_offset, res_bbox.width, res_bbox.height))
        print('score %s' % res_bbox.confidence_level)
        counter += 1
        t_2 = time.time()
        print('time for OS iteration is %s seconds, fps is %s' % (t_2 - t_1, 1/(t_2 - t_1)))



    tracker.freeze()  # starting flip (saving last image)
    tb_total = 0
    for f in range(Bytetrack_frames):
        tb_0 = time.time()
        tracker.img_after_flip, tracker.frame_id_flip = tracker.get_last_img()
        img_ref = tracker.img_after_flip
        orig_imgsz = img_ref.shape
        print('image ref shape is :',str(img_ref.shape))
        if not opt.ref_point:
            i_ref , j_ref = int(imgsz[0]/2) , int(imgsz[1]/2)
            ref_point = [i_ref, j_ref]
        img_after_flip = img_reshape(crop_ref, ref_point, model, img_ref, imgsz, orig_imgsz, stride, pt)
                #img_after_flip = img_after_flip.cpu().detach().numpy()
                #print(cv2.imwrite('frame.jpg', img_after_flip))

        tb_1 = time.time()
        print('model image shape' + str(img_after_flip.shape))
        preds = model(img_after_flip, augment= opt.augment, visualize=False)
        tb_2 = time.time()
        preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=100)
        print('preds are : ' + str(preds))
        tb_3 = time.time()
        for i, det in enumerate(preds):
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img_after_flip.shape[2:], det[:, :4], img_ref.shape).round() #TODO - check if scalse coords instead!

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                Online_targerts = bytetracker_handle(opt, det, xywhs, imgsz, bytetracker)
                multiple_num = float(opt.tracker_lowfps)
                online = Online_handle(det, opt, Online_targerts, multiple_num)
        new_bbox = Euclidean_algo(preds, [i_ref,j_ref], img_after_flip, orig_imgsz, device=device )

        tracker.last_bbox = new_bbox # updating tracker bbox after flip !
        tb_4 = time.time()
        print('time for image load is %s seconds, fps is %s' % (tb_1 - tb_0,  1/(tb_1 - tb_0)))
        print('time for model is %s seconds, fps is %s' % (tb_2 - tb_1,  1/(tb_2 - tb_1)))
        print('time for NMS is %s seconds, fps is %s' % (tb_3 - tb_2,  1/(tb_3 - tb_2)))
        print('time for scale & filter is %s seconds, fps is %s' % (tb_4 - tb_3,  1/(tb_4 - tb_3)))
        print('time for ByteTracker iteration is %s seconds, fps is %s' % (tb_4 - tb_0,  1/(tb_4 - tb_0)))
        tb_total += 1/(tb_4 - tb_0)
    print('avarage fps for ByteTracker is %s' %(tb_total/Bytetrack_frames))

    tracker.flip = 1
    tracker.frame_after_flip = 1
    # END

    tracker.msg = 'tracking'
    #while not tracker.msg.__eq__('stop'):
    while res:
        t_5 = time.time()
        res = tracker.acquire() # attack mode
        tracker.frame_after_flip = 0
        if (res):
            res_bbox = res['bbox']
            frame_id = res['frame_id']
            print('counter of trakcing: %s' % counter)
            print('bbox: %s , %s , %s, %s' % (res_bbox.x_offset, res_bbox.y_offset, res_bbox.width, res_bbox.height))
            print('score %s' % res_bbox.confidence_level)
            counter += 1
        t_6 = time.time()
        print('time for OS iteration is %s seconds, fps is %s' % (t_6 - t_5,  1/(t_6 - t_5)))






if __name__ == "__main__":
    from utils.general import BoundingBox

    # ArgParser

    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('--tracker-name', type=str, default='ostrack', help='Name of tracking method.')
    parser.add_argument('--tracker-param', type=str, default='vitb_256_mae_ce_32x4_ep300', help='Name of parameter file.')
    parser.add_argument('--videofile', type=str, default='/home/shai/shai/rocx/soy/SteepDive_Head.mp4', help='path to a video file.')
    parser.add_argument('--track-mode', type=str, default='attack', help='attack or hover mode')
    parser.add_argument('--optional-box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save-results', type=int, default=1, help='Save bounding boxes')
    parser.add_argument('--use-half', type=int, default=0, help='Save bounding boxes')
    parser.add_argument('--use-trt', type=int, default=0, help='Save bounding boxes')
    parser.add_argument('--flip-point' , type=int, default=2, help='point of flipping')
    parser.add_argument('--init_bbox' , type=list, default=[895, 743, 41, 49], help='initial box for OS tracker')
    parser.add_argument('--ref_point', default=0, help='referance point')
    parser.add_argument("--crop-ref", type=int, default= 0, help='Whether to crop the image according to referance point or not')

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

    parser.add_argument('--use-bytetracker', action='store_true', default=True , help='use byte tracker or deepsort')
    parser.add_argument('--add-yolobboxes', action='store_true', default=False , help='add a bbox for yolo detections')
    parser.add_argument('--tracker-lowfps', type=int, default=1 , help='multiple x4 bbox size to tracker')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    
    main(opt)
    


