# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import subprocess
import sys
sys.path.insert(0, './yolov5_v7')
from tqdm import tqdm
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from yolov5ben.models.experimental import attempt_load
# #from yolov5ben.utils.downloads import attempt_download
# from yolov5ben.models.common import DetectMultiBackend
# from yolov5ben.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
# from yolov5ben.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
#                                   check_imshow, xyxy2xywh,xyxy2xywhn, xywh2xyxy, increment_path, strip_optimizer, colorstr)
# from yolov5ben.utils.torch_utils import select_device, time_sync
# from yolov5ben.utils.plots import Annotator, colors, save_one_box

from yolov5_v7.models.experimental import attempt_load
from yolov5_v7.models.common import DetectMultiBackend
from yolov5_v7.utils.dataloaders import LoadImages , LoadStreams, VID_FORMATS # changed from datasets to dataloaders
from yolov5_v7.utils.general import LOGGER, check_img_size, non_max_suppression, scale_boxes, check_imshow, \
    xyxy2xywh, xyxy2xywhn, xywh2xyxy, increment_path , strip_optimizer, colorstr
from yolov5_v7.utils.torch_utils import select_device, time_sync
from yolov5_v7.utils.plots import Annotator, colors, save_one_box
from util_for_api import bbox_to_angles
from pedestal.pedestal_init import start_pedestal
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from datetime import datetime
from glob import glob
FILE = Path(__file__).resolve()
#FILE = Path(r'Z:\deepsort_yolov5\deepsort_ben\track.py') # FOR DEBUG = DELETE IT AFTER
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
logging.basicConfig(filename='experiment_pc_track_logs_'+ str(datetime.now()) + '.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s -%(message)s')
logger = logging.getLogger(__name__)


def tlwh_to_xyxy(bbox_tlwh, widht, hight):
    x, y, w, h = bbox_tlwh
    x1 = max(int(x), 0)
    x2 = min(int(x + w), widht - 1)
    y1 = max(int(y), 0)
    y2 = min(int(y + h), hight - 1)
    return x1, y1, x2, y2


def tlwh2xywh(x):
    x_center = x[0] + (x[2] / 2)  # x center
    y_center = x[1] + (x[3] / 2)  # y center
    width = x[2]  # width
    height = x[3]  # height
    return x_center, y_center, width, height


def truncate(n, decimals=7):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def is_detected(detected, conf_tmp, cls_tmp, theshold):
	sec = 0
	for i in range(len(conf_tmp)):
		if (conf_tmp[i] > theshold and cls_tmp[i] == 0):
			sec = 1
	return [sec, detected[0], detected[1]]

def make_noise():
    cmd = "gst-launch-1.0 filesrc location=Drone_record.wav ! wavparse ! audioconvert ! audioresample ! alsasink device=hw:0"
    x =subprocess.Popen(cmd, shell=True)


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
        project, exist_ok, update, save_crop, exp_name = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop, \
        opt.name
    # We added source for mapping the webcam!!!
    webcam = source == '0' or source == '2' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    detected = [0, 0, 0]

    if opt.magi_alg:
        import requests  # FOR MAGI
    if opt.raw_recording:
        opt.save_all_images = True
        print("opt.save_all_images = True, because raw_recording is TRUE")

    ##BYTE
    #sys.path.insert(0, '../../ByteTrack/')
    from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
    tracker = BYTETracker(opt)
    ###
    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    # if type(yolo_model) is str:  # single yolo model
    #     exp_name = yolo_model.split(".")[0]
    # elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
    #     exp_name = yolo_model[0].split(".")[0]
    # else:  # multiple models after --yolo_model
    #     exp_name = "ensemble"
    # # exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]

    #save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    save_dir = (Path(project) / exp_name)
    if opt.use_bytetracker:
        save_dir = save_dir / 'bytetracker'
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir  # increment run if project name exists  # BEN

    # Load model
    # took it from yolov5_v7 val.py dnn = false
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn, fp16=half)
    #model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    real_imgsz = imgsz
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        #show_vid = check_imshow()
        pass
    # Dataloader
    if webcam:
        show_vid = True # check_imshow() retrun False
        #print(check_imshow())
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    counter = 0
    file_name_counter = 0
    time_now = str(datetime.now())  # date and time for image and text file names
    cout_swimmers_id = {}
    start_time = time_sync()
    frame_counter = 1
    t1 = 0
    fps_p = 0
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(tqdm(dataset, total=len(dataset))): #webcam len = 1
        t0 = time_sync()
        if t1:
            print("")
            print((t0 - t1))
            fps_p = (1.0 / (t0 - t1))
            print('fps_p: %f' % fps_p )
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS) or (os.listdir(source)[0].endswith(VID_FORMATS)):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            txt_path = str(save_dir / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
				#det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                #print("det before: %s" % det[:, 0:7])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()

                if opt.use_bytetracker:  ### BYTE Tracker!
                    multiple_num = float(opt.tracker_lowfps)
                    if opt.tracker_lowfps > 1: #multiple_det_for_low_fps
                        xywhs[:,2] *= multiple_num
                        xywhs[:,3] *= multiple_num
                        xyxy_multiple = xywh2xyxy(xywhs)
                        det[:,0:4] = xyxy_multiple
                        #print("input for bytetracker det %s" % det[0, 0:4])
                        online_targets = tracker.update(det.cpu().numpy(), imgsz, imgsz)
                        #return it back
                        xywhs[:, 2] /= multiple_num
                        xywhs[:, 3] /= multiple_num
                        xyxy_multiple = xywh2xyxy(xywhs)
                        det[:, 0:4] = xyxy_multiple
                    else:
                        #print("input for bytetracker det %s" % det[0, 0:4])
                        online_targets = tracker.update(det.cpu().numpy(), imgsz, imgsz)
                    #print(online_targets)
                    online = []
                    if opt.gilui_proj:
                        score_det = []
                        cls_det = []
                    for _, t in enumerate(online_targets):
                        #print("found bytetracker")
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
                        if opt.gilui_proj:
                            score_det.append(score)
                            cls_det.append(cls)
                        for one_det in det:
                            one_conf= one_det[4]
                            one_clss = one_det[5]
                            if round(float(one_conf),2) == round(float(score),2) and int(one_clss) == int(cls):
                                x1, y1, x2, y2 = one_det[0:4].cpu()#, one_det[1] , one_det[2] , one_det[3]
                                #print("%s"%(one_det[0:4]))
                        online.append(np.array([x1, y1, x2, y2, tid, cls, score]))
                        # END
                        #online.append(np.array([x1, y1, x2, y2, tid, cls, score]))
                    if len(online) > 0:
                        online = np.stack(online, axis=0)
                    outputs[i] = online
                else: # DEEP SORT
                    outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                t5 = time_sync()
                dt[3] += t5 - t4

                if opt.gilui_proj:
                    detected = is_detected(detected, score_det, cls_det, theshold=opt.track_thresh)
                    if (detected[0] == 1 and detected[1] == 1 and detected[2] == 1):
                        print('\nMake Noise...')
                        make_noise()

                best_conf = 0.0
                best_id = -1
                best_conf_index = -1
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        if int(cls) == 0 and float(conf) > best_conf:
                            best_conf, best_id, best_bboxes = conf, id , bboxes
                            best_conf_index = j
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))  # BEN DONT SAVE BBOX

                        # if c == 2:
                        #     if not cout_swimmers_id:
                        #         cout_swimmers_id[id] = 1
                        #     elif id not in cout_swimmers_id.keys():
                        #         cout_swimmers_id[id] = 1
                        #     else:
                        #         cout_swimmers_id[id] += 1
                            # for key in cout_swimmers_id.keys():
                            #     print('swimmers id: %d counter %d' % (key, cout_swimmers_id[key] ))
                        if save_txt:
                            #print("save_txt = True")
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            w, h = real_imgsz
                            # Write MOT compliant results to file
                            try:
                                txt_path = str(Path(save_path) / Path(Path(path).name).with_suffix('.txt'))

                                if '.' in str(Path(save_path)): # if its video so do not make a dicrectory and change path of file
                                    txt_path = str(Path(save_path).parents[0] / Path(Path(path).name).with_suffix('.txt'))
                                else:
                                    Path(save_path).mkdir(exist_ok=True)
                            except Exception as e:
                                pass
                                #print(e)
                                print("HERE IS THE ERROR, DO NOT WORRY ITS STILL WORKING.")
                            if opt.write_format == 'MOT':
                                print("MOT Format")
                                with open(txt_path, 'a') as f:
                                    #f.write(('%g ' * 10 + '\n') % (id, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, i, frame_idx + 1))
                                    #c = class ,
                                    f.write(('%g, ' * 11 + '%g' + '\n') % (frame_idx + 1, id, bbox_left / w, bbox_top / h, bbox_w / w, bbox_h / h, -1, c, -1, -1, -1, conf )) #MOT CHALLANGER FORMAT BEN
                                    #print('h')
                                    # f.write(('%g, ' * 12 + '\n') % (frame_idx + 1, c, id, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, -1, conf))  # FORMAT BEN FROM EMAIL
                            elif opt.write_format == 'Yolo':
                                #print("Yolo Format")
                                box = xyxy2xywhn(bboxes.reshape(1,-1), w, h).flatten()

                                txt = (c, truncate(float(box[0])), truncate(float(box[1])), truncate(float(box[2])), truncate(float(box[3])), conf,  str(datetime.now()),str(fps_p), str(frame_counter))
                                s = 9
                                ### BEN
                                if show_vid:
                                    new_folder = Path(str(Path(save_path).parents[0] / Path(time_now)))
                                    new_folder.mkdir(exist_ok=True)
                                    txt_path = str(new_folder) + '/' + str(file_name_counter) + '.txt'
                                    #txt_path = str(Path(txt_path).parents[0] / Path(time_now)) + '_' + str(file_name_counter) + '.txt'
                                #print("txt_path %s" % txt_path)
                                #txt_file_name = path.split('/')[-1].split('.')[0]
                                #txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'
                                ###
                                with open(txt_path, 'a') as f:
                                    f.write('%g %g %g %g %g %g %s %s %s\n' % txt)
                                    logging.info(txt)
                            else:
                                raise ValueError(f'write format not valid: {opt.write_foramt}')
                            # if save_vid or save_crop or show_vid:  # Add bbox to image
                            #     c = int(cls)  # integer class
                            #     label = f'{id:0.0f} {names[c]} {conf:.2f}'
                            #     annotator.box_label(bboxes, label, color=colors(c, True))
                            #     if save_crop:
                            #         txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                            #         save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                    if opt.magi_alg:
                        # still inside if len of output > 1, TAKING the best conf (same cls)
                        # print('outputs: %s' % outputs)
                        # print('best: conf , id : %s , %s' % (best_conf, best_id))
                        # print('best_conf_index : %s ' % best_conf_index)
                        # print('best_bboxes: %s' % best_bboxes)
                        wd = best_bboxes[2] - best_bboxes[0]
                        he = best_bboxes[3] - best_bboxes[1]
                        x_tl = best_bboxes[0]
                        y_tl = best_bboxes[1]

                        img_width = im0.shape[1]
                        img_height = im0.shape[0]
                        #print("x_tl, y_tl, wd, he, img_width, img_height: %s,%s,%s,%s,%s,%s" % (x_tl, y_tl, wd, he, img_width, img_height))
                        try:
                            az, el = bbox_to_angles(x_tl, y_tl, wd, he, img_width, img_height)
                            print('az: %s  ,  el: %s' %(az,el))
                            az_file = Path(txt_path).parents[0] / 'az_el.txt'
                            with open(az_file, 'a') as f:
                                f.write("best id: %s, az: %s , el: %s, datetime: %s \n" % (best_id,az,el, str(datetime.now())))
                                logging.info("best id: %s, az: %s , el: %s, datetime: %s , counter_frames %s\n" % (best_id,az,el, str(datetime.now()), str(frame_counter)))
                            url = 'http://localhost:8000/move_pedestal/'
                            headers = {
                                'accept': 'application/json',
                                'Content-Type': 'application/json',
                            }
                            data = {
                                'axis_x': az,
                                'axis_y': el
                            }
                            response = requests.post(url, headers=headers, json=data)
                            print('RESPONSE FROM REST API: %s' % response)
                        except Exception as e:
                            print(e)

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort_list[i].increment_ages()
                LOGGER.info('No detections')

            #cls_name = ['Drone', 'AirPlane', 'UFO', 'AirPlane_w_lights', 'Baloons', 'Drone_w_lights', 'Single_front_lights', 'other']
            if opt.add_yolobboxes:  # det[:,0:4]= xyxy , det[:, 4]= confg , det[:, 5]= class
                for val in det:
                    x1 = int(val[0])
                    y1 = int(val[1])
                    x2 = int(val[2])
                    y2 = int(val[3])
                    conf = round(float(val[4]), 2)
                    cls = int(val[5])
                    color = colors(cls)
                    im0 = cv2.rectangle(img=im0, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2)
                    cv2.putText(im0, names[cls] + '_' + str(conf), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cp_img = cv2.resize(im0, (1920,1080))
                cv2.imshow(str(p), cp_img)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = fps_p, im0.shape[1], im0.shape[0] # if it looks bad change it to 30 or 60 fps
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    if opt.out_video:
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        #vid_writer[i] = cv2.VideoWriter(save_path, -1, fps, (w, h))
                if opt.out_video:
                    vid_writer[i].write(im0)
                else:
                    if opt.save_all_images or (det is not None and len(det)):  # SAVE IMAGES ONLY WHILE WE HAVE DETECTIONS
                        #print(str(Path(save_path).parents[0]/Path(path).name))
                        # if show_vid: # WRITE FRAMES FROM CAMERA TO FOLDER
                        #     new_folder = Path(str(Path(save_path).parents[0] / Path(time_now)))
                        #     new_folder.mkdir(exist_ok=True)
                        #     new_path = str(new_folder) + '/' + str(file_name_counter) + '.jpg'
                        # else:
                        #     new_path = str(Path(save_path)/Path(path).name)
                        new_folder = Path(str(Path(save_path).parents[0] / Path(time_now)))
                        new_folder.mkdir(exist_ok=True)
                        new_path = str(new_folder) + '/' + str(file_name_counter) + '.jpg'
                        cv2.imwrite(new_path, im0)
                        file_name_counter += 1
                        if opt.raw_recording:
                            recording_path = str(new_folder) + '/' + 'recording' + '/' + str(file_name_counter) + '.jpg'
                            Path(str(new_folder) + '/' + 'recording').mkdir(exist_ok=True)
                            cv2.imwrite(recording_path, imc)
        #print("average time per frame : %s " %  ((time_sync()-start_time)/frame_counter) )
        frame_counter += 1

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(glob(save_path + r'/*.txt'))} tracks saved to {save_path}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_path)}{s}")
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)
		
    # count_all_swimmers = 0
    # for key in cout_swimmers_id.keys():
    #     print('swimmers id: %d counter %d' % (key, cout_swimmers_id[key] ))
    #     count_all_swimmers += cout_swimmers_id[key]
    # print('total frames: %d , and total ids %d' % (count_all_swimmers, len(cout_swimmers_id.keys())))
	
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_ibn_x1_0_MSMT17')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[3840,2160], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--out-video', action='store_true', default=False , help='export video')
    parser.add_argument('--write-format',type=str, default='Yolo', help='detections writing format')
    parser.add_argument('--raw-recording', action='store_true', default=False, help='output raw images')
    parser.add_argument('--magi-alg', action='store_true', default=False, help='turn on magi algo to send best result with "request"')
    parser.add_argument('--gilui-proj', action='store_true', default=False, help='turn on magi algo to send best result with "request"')
    parser.add_argument('--save-all-images', action='store_true', default=False, help='save all images and not only detected')


    #### BYTE TRACKER
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min_track_thresh", type=float, default=0.1, help='threshold for minimum assignment')

    parser.add_argument('--use-bytetracker', action='store_true', default=False , help='use byte tracker or deepsort')
    parser.add_argument('--add-yolobboxes', action='store_true', default=False , help='add a bbox for yolo detections')
    parser.add_argument('--tracker-lowfps', type=int, default=1 , help='multiple x4 bbox size to tracker')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print(f"use ByteTracker: {opt.use_bytetracker}")
    print(f"use rectangle Yolo BBox: {opt.add_yolobboxes}")
    print(f"use multiple tracker for low fps: {opt.tracker_lowfps}")

    with torch.no_grad():
        detect(opt)
