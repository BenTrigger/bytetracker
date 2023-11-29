import torch
import torchvision.ops as ops
import numpy as np
import time

from config.project_config import init_params
from tracker.kalman_filter import KalmanFilter
from tracker.kalman_track import KLtrack
from trt.loading_trt import inference_engine, set_trt, TRT_Preprocessor
from utils.general import BoundingBox
from utils.methods import hann2d, bboox2list, clip_box_batch, get_search_area, get_front_back,\
    get_corner_speed, is_contained, map_box_back_batch, is_correct_ave, calc_iou_overlap, cal_top_k_bbox
from utils.processing_utils import sample_target, generate_mask_cond
from utils.misc import SysState, SysScore, SkippingMode
from utils.reg import resize, in_registration
# from rocx.tracker.msg import TrackerStatus, Orientation
import cv2


# import logging

# logging.basicConfig(filename='lite_tracker_ORG.log',
#                     level=logging.DEBUG,
#                     format='%(levelname)s - %(message)s')

class LiteTracker:
    def __init__(self, config_file="params.yaml"): # status_callback: Callable[[TrackerStatus]):
        # one more option is to read params from yaml fime and get path
        # self.logger = logging.getLogger(self.__class__.__name__)
        # self.logger.info('Init LiteTracker')
        self.params = init_params(config_file, 1)  # read in here.
        #for field_name, field_value in self.params.__dict__.items():
            # self.logger.info(f'{field_name}, {field_value}')
        self.reg_params = init_params(config_file, 2)  # read in here.
        #for field_name, field_value in self.reg_params.__dict__.items():
            # self.logger.info(f'{field_name}, {field_value}')
        self.trt_args = set_trt(self.params.TRT_FILE_PATH)
        # self.logger.info(f'trt_args: {self.trt_args}')
        self.init_hyper_params(config_file)
        self.reset()

        # motion constrain
        self.feat_sz = self.params.SEARCH_SIZE // self.params.MODEL_BACKBONE_STRIDE
        # self.logger.info(f'feat_sz: {self.feat_sz}')
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True)
        # self.logger.info(f'output_window: {self.output_window}')
        self.preprocessor = TRT_Preprocessor()
        # self.logger.info(f'preprocessor: {self.preprocessor}')

        # delete cam param after integration
        self.cam = None
        self.cam_listener = 0
        self.cam_frame_counter = 0
        # color of rect
        self.flip = 0
        self.frame_after_flip = 0
    # region API
    def warmup(self, steps=1000):
        # TO DO BEN AND SHAI: change it to tensorRT !
        # self.logger.debug(f'warmup')
        z_sz = self.params.TEMPLATE_SIZE   # 128 h
        # self.logger.info(f'z_sz: {z_sz}')
        x_sz = self.params.SEARCH_SIZE  # 256 = w
        # self.logger.info(f'x_sz: {x_sz}')
        template = np.random.randn(1, 3, z_sz, z_sz)  # to(device)
        # self.logger.info(f'template: {template}')
        search = np.random.randn(1, 3, x_sz, x_sz)
        # self.logger.info(f'search: {search}')
        trt_input = [template, search]
        for i in range(steps):  # BEN CHECK IT - NEW METHOD TO WARM UP WITH TENSORRT
            _ = inference_engine(trt_input, self.trt_args) # equal to forward in network
        # self.logger.debug(f'warmup DONE')
        return True

    def initialize(self, bbox: BoundingBox, frame_id):  # info: dict):
        # self.logger.debug(f'initialize')
        bbox_list = bboox2list(bbox)
        # self.logger.info(f'bbox_list: {bbox_list}')
        self.last_bbox = bbox_list
        # self.logger.info(f'self.last_bbox: {self.last_bbox}')
        self.state = bbox_list
        # self.logger.info(f'self.state: {self.state}')
        # DELETE IT - BEN
        #self.cap = cv2.VideoCapture(r"C:\Users\benew\Downloads\4kroad.mp4")
        #self.cap = cv2.VideoCapture(r"../../ben_trt/tracker_poc-rocx_15_fps_final_version_before_train/vid0.avi")
        #self.cap = cv2.VideoCapture(r"../../yovel0.avi")

        #einat data
        #self.cap = cv2.VideoCapture(r"../../data_from_einat/reddrone-1444_.mp4")
        #self.cap = cv2.VideoCapture(r"/work/nvidia/36_last_version_21_10_23/tracker_engine/data/20231013_125602.mp4")
        self.cap = cv2.VideoCapture(r"/workspace/36_last_version_21_10_23/Tracker_OS_Byte/20231013_125714.mp4")
        #self.cap = cv2.VideoCapture(r"../../data_from_einat/matrice600_1436_.mp4")
        #self.cap = cv2.VideoCapture(r"../../data_from_einat/matrice600-1414_640_End2.mp4")
        # self.cap = cv2.VideoCapture(r"../../data_from_einat/matrice600-1414_640_Cross.mp4")
        #self.cap = cv2.VideoCapture(r"/workspace/ben_trt/TLV_4K.mp4")

        image = self.get_img_by_frame_id(frame_id)  # need a real image to test it !
        if image is None:
            raise Exception("The image is empty.")
        #if not image.isValid():
        #    raise Exception("The image is not valid.")
        cv2.imwrite('image_exps/frame_00000.jpg',image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # self.logger.info(f'Image was received using get_img_by_frame_id: {image.shape}')
        # TILL HERE

        # forward the template once
        self.forward_template(image, self.state)

        # save states
        score = 1.
        self.is_init_obj_shape = True
        self.is_stable = False
        self.template_update_info = [{
            'frame_counter': self.frame_counter,
            'bbox': bbox_list,
            'steps': 0
        }]
        # self.logger.info(f'self.template_update_info: {self.template_update_info}')

        self.old_image = {
            'image': [image] * self.seq_length,
            'bbox': [bbox_list] * self.seq_length,
            'id': 0,
            'score': torch.tensor([score] * self.seq_length)
        }
        # self.logger.info(f'old_image: {self.old_image}')
        self.score_avg = score
        # self.logger.info(f'score_avg: {self.score_avg}')

        self.KLtrack = KLtrack(self.state, score)
        # self.logger.info(f'KLtrack: {self.KLtrack}')
        # self.logger.info(f'kalman_filter: {self.kalman_filter}')
        # self.logger.info(f'frame_counter: {self.frame_counter}')
        self.KLtrack.activate(self.kalman_filter, self.frame_counter)

        out = {"target_bbox": bbox_list}
        # self.logger.info(f'out: {out}')
        # Exponential Moving Average
        self.ema = np.array(self.state)
        # self.logger.info(f'ema: {self.ema}')
        self.sma = np.average(np.array(self.old_image['bbox'])[:, 2] * np.array(self.old_image['bbox'])[:, 3])
        # self.logger.info(f'sma: {self.sma}')
        self.skipping = SkippingMode('by_Score_Located')

        # self.logger.debug(f'initialize DONE')
        return out

    def acquire(self):
        """
        Acquire target
        Start "hover" tracking
        :return: True if success
        """
        # self.logger.debug(f'acquire')
        if not self.frame_after_flip:
            img, self.frame_id = self.get_last_img()
        else:
            img, self.frame_id = self.img_after_flip, self.frame_id_flip
                    # self.logger.info(f'img, self.frame_id: {img.shape, self.frame_id}')
        if img is None:
            return_value = None
        else:
            # need to fill INFO param. ! BEN
            bbox = self.track(image=img)
            return_value = {
                'bbox': bbox,
                'frame_id': self.frame_id
            }
        # self.logger.info(f'return_value: {return_value}')
        # self.logger.debug(f'acquire DONE')
        return return_value

    def freeze(self):
        """
        Freeze target
        Saving last frame because of the camera staring to flip
        :return: True if success
        """
        self.start_flip_time = time.time()

        self.bbox_before_flip = self.last_bbox
        # getting last frame from camera number 0
        self.img_before_flip = self.get_img_by_frame_id(self.frame_id) # make sure we change it, so we get image only.
        self.img_before_flip = resize(self.img_before_flip, scale_factor=self.reg_params.scale_factor)
        return True

    def reacquire(self):
        """
        1. Camera flipped, We are taking the new first image to register.
        2. and start new tracking "attacking"
        3. Call it only once !
        Reacquire target
        :return: True if success
        """
        # getting last frame from camera number 1
        self.cam_listener = 1
        self.cam = None
        self.img_after_flip, cur_frame_id = self.get_last_img()
        self.last_bbox = in_registration(self.img_after_flip, self.img_before_flip, self.bbox_before_flip, coff=self.reg_params.reg_coff, disp=True)

        total_flip_time = time.time() - self.start_flip_time

        # ToDo add Kalman prediction

        print('total flip time: %s' % total_flip_time)

        self.drone_mode = "attack"

        x1, y1, w, h = self.last_bbox
        bbox = BoundingBox(x1, y1, w, h, 1)  # there is no conf calculation at this moment

        return_value = {
            'bbox': bbox,
            'frame_id': cur_frame_id
        }
        return return_value

    def reset(self):
        """
        Reset tracker params and
        wait for initilaize & aquire call ( to start tracking )
        :return: True if success
        """
        self.kalman_filter = KalmanFilter()
        self.KLtrack = None
        self.system_state = SysState()
        self.system_score = SysScore()
        self.frame_counter = 0
        self.drone_mode = 'hover'  # hover/attack
        self.state = None
        self.start_flip_time = None
        self.bbox_before_flip = None
        self.start_flip_time = None
        self.img_before_flip = None
        self.img_after_flip = None

        return True

    # endregion

    # region initialize and tracking mode

    def track(self, image, k: int = 5):  # 'hover'
        # self.logger.debug(f'track')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BEN
        # self.logger.info(f'frame_counter: {self.frame_counter}')
        if self.frame_counter == 1:
            self.old_image['init_gt'] = [image, self.state]
            # self.logger.info(f'old_image: {self.old_image}')


        self.frame_counter += 1
        # self.logger.info(f'frame_counter: {self.frame_counter}')
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.SEARCH_FACTOR,
                                                                output_sz=self.params.SEARCH_SIZE)  # (x1, y1, w, h)
        # self.logger.info(f'x_patch_arr: {x_patch_arr}')
        # self.logger.info(f'resize_factor: {resize_factor}')
        # self.logger.info(f'x_amask_arr: {x_amask_arr}')

        search, search_mask = self.preprocessor.process(x_patch_arr, x_amask_arr)
        # self.logger.info(f'search: {search}')
        out_dict = inference_engine([self.template[0], search], self.trt_args) # equal to forward in network
        # # self.logger.info(f'out_dict: {out_dict}')

        crop2img = self.params.SEARCH_SIZE / resize_factor
        # self.logger.info(f'crop2img: {crop2img}')

        score_map_ctr = out_dict['score_map']
        # # self.logger.info(f'score_map_ctr: {score_map_ctr}')

        # add hann windows
        if self.system_state.get_state() == 'track':
            # self.logger.info(f'system_state.get_state(): {self.system_state.get_state()}')
            score_map_ctr *= self.output_window
            # # self.logger.info(f'score_map_ctr: {score_map_ctr}')

        self.KLtrack.predict()
        kalman_bbox = self.KLtrack.tlwh
        # self.logger.info(f'kalman_bbox: {kalman_bbox}')

        self.check_sma_ok()
        final_box, score, candidates = self.get_final_bbox(kalman_bbox,
                                                           score_map_ctr,
                                                           out_dict['size_map'],
                                                           out_dict['offset_map'],
                                                           return_score=True,
                                                           k=k,
                                                           crop2img=crop2img,
                                                           shape=image.shape)
        if self.frame_after_flip:
            final_box = torch.tensor(self.last_bbox)
        # self.logger.info(f'final_box: {final_box}')
        # self.logger.info(f'score: {score}')

        # Exponential Moving average
        #print('final box ' + str(final_box))
        self.ema = final_box * self.ema_alpha + (1 - self.ema_alpha) * self.ema
        # self.logger.info(f'ema: {self.ema}')

        self.state = final_box.tolist()
        # self.logger.info(f'state: {self.state}')
        self.template_update_info[-1]['steps'] += 1
        # self.logger.info(f'template_update_info: {self.template_update_info}')

        # check the final box and update system state
        self.update_sys_state(score, final_box)
        self.update_sys_score(score)
        self.is_hiding()

        # update kalman
        KL_update = score > self.tracking_mode_thresh and self.system_score.get_score() != 'lost'
        # self.logger.info(f'KL_update: {KL_update}')

        if self.drone_mode == 'hover':
            # self.logger.info(f'self.drone_mode == "hover"')
            if (self.system_state.get_state() == 'search' and self.system_score.get_score() == 'lost'):  # GOTO kalman
                # self.logger.info(f"self.system_state.get_state() == 'search' and self.system_score.get_score() == 'lost'")
                self.state = kalman_bbox
                # self.logger.info(f'state: {self.state}')
                self.is_kalman_box = True
                self.system_state.set_state('acq')
                # self.logger.info(f'system_state: {self.system_state.get_state}')
            KL_update = KL_update and not self.is_hid and self.frame_counter - self.last_hid > self.update_after_hid
            # self.logger.info(f'KL_update: {KL_update}')
        if KL_update:
            # self.logger.info(f'KLtrack.update...')
            self.KLtrack.update(self.state, self.frame_counter)
        else:
            # self.logger.info(f'KLtrack.mean...')
            self.KLtrack.mean[6], self.KLtrack.mean[7] = 0, 0

        # update the template
        min_steps_to_update_tempalte = self.min_steps_to_update_template[self.drone_mode]
        # self.logger.info(f'min_steps_to_update_tempalte: {min_steps_to_update_tempalte}')
        Ciou = 0
        if self.drone_mode == 'hover':
            kalman_tensor = torch.tensor(kalman_bbox, device=final_box.device).unsqueeze(0)
            # self.logger.info(f'kalman_tensor: {kalman_tensor}')
            kalman_xyxy = ops.box_convert(kalman_tensor, 'xywh', 'xyxy')
            # self.logger.info(f'kalman_xyxy: {kalman_xyxy}')
            final_xyxy = ops.box_convert(final_box.reshape(1, -1), 'xywh', 'xyxy')
            # self.logger.info(f'final_xyxy: {final_xyxy}')
            Ciou = ops.complete_box_iou(final_xyxy, kalman_xyxy)
            # self.logger.info(f'Ciou: {Ciou}')
            # to evoid divergence
            min_steps_to_update_tempalte = self.min_steps_to_update_template[self.drone_mode][0] if score < 0.6 else \
            self.min_steps_to_update_template[self.drone_mode][1]
            # self.logger.info(f'min_steps_to_update_tempalte: {min_steps_to_update_tempalte}')

        if self.system_state.get_state() == 'track' and not self.is_hid and self.frame_counter - self.last_hid > self.update_after_hid:
            self.add_img_to_template_array(image, score, score_thresh=self.add_img_to_template_score_thresh)
        self.check_and_update_template(thresh=0, steps_to_update=min_steps_to_update_tempalte, score=Ciou)

        self.prev_is_kalman_box = False
        if self.is_kalman_box:
            # self.logger.info('is_kalman_box')
            self.prev_is_kalman_box = True


        # region debug
        # if self.debug:
        #     # iou_score = calc_iou_overlap(final_box.reshape(1,-1), torch.tensor([self.last_bbox], device=final_box.device).reshape(1,-1)).item()
        #     kalman_tensor = torch.tensor(kalman_bbox, device=final_box.device).unsqueeze(0)
        #     kalman_xyxy = ops.box_convert(kalman_tensor, 'xywh', 'xyxy')
        #     final_xyxy = ops.box_convert(final_box.reshape(1, -1), 'xywh', 'xyxy')
        #     Ciou = ops.complete_box_iou(final_xyxy, kalman_xyxy)
        #     from tracking.show_results import Colors
        #     colors = Colors()
        #     image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     labels = [str(score)[:6], 'kalman', 'search']
        #
        #     # labels = [str(score)[:6],'kalman','search']
        #     def get_search_area(target_bb, search_area_factor):
        #         x, y, w, h = target_bb
        #         # Crop image
        #         crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
        #
        #         if crop_sz < 1:
        #             raise Exception('Too small bounding box.')
        #
        #         x1 = round(x + 0.5 * w - crop_sz * 0.5)
        #         y1 = round(y + 0.5 * h - crop_sz * 0.5)
        #
        #         return [x1, y1, crop_sz, crop_sz]
        #
        #     search_bbox = get_search_area(self.state, self.params.SEARCH_FACTOR)
        #
        #     if len(candidates) > 2:
        #         nms_bboxs, max_rag, final_boxes = candidates
        #         for i, box in enumerate(final_boxes):
        #             if i != nms_bboxs[max_rag]:
        #                 x1, y1, w, h = final_boxes[i]
        #                 cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255),
        #                               thickness=2)
        #
        #     for i, box in enumerate([final_box.cpu().numpy(), kalman_bbox, search_bbox]):
        #         # for i, box in enumerate([final_box.cpu().numpy(), kalman_bbox, search_bbox]):
        #         # if i !=1:
        #         #     continue
        #         label = labels[i]
        #         x1, y1, w, h = box
        #         cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=colors(i), thickness=2)
        #         _, ht = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]  # text width, height
        #         outside = y1 - ht >= 3
        #         cv2.putText(image_BGR,
        #                     label, (int(x1), int(y1 - 2) if outside else int(y1 + ht + 2)),
        #                     0,
        #                     1 / 3,
        #                     colors(i),
        #                     thickness=1,
        #                     lineType=cv2.LINE_AA)
        #     # TOP LEFT DATA
        #     stats = {
        #         'score: ': round(score, 3),
        #         'Ciou_score: ': round(Ciou.item(), 3),
        #         'sma_frac: ': round(self.frac, 3),
        #         'stable: ': 'yes' if self.is_stable else 'no',
        #         'shape: ': 'yes' if self.is_init_obj_shape else 'no',
        #         'state: ': self.system_state.get_state(),
        #         'user score: ': self.system_score.get_score(),
        #         'is_hid: ': self.is_hid,
        #     }
        #     for i, (key, val) in enumerate(stats.items()):
        #         y = 50 * (i + 1)
        #         cv2.putText(image_BGR,
        #                     key + str(val), (50, y),
        #                     # key + str(val)[:5], (50, y),
        #                     0,
        #                     1,
        #                     colors(15),
        #                     thickness=4,
        #                     lineType=cv2.LINE_AA)
        #     save_dir = self.save_dir + '/' + info['seq_name'].split(' ')[0]
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_path = save_dir + ('/%04d.jpg' % self.frame_counter)
        #     cv2.imwrite(save_path, image_BGR)
        # endregion

        is_located = self.skipping.check_if_located(self.state, self.last_bbox)
        # self.logger.info(f'check_if_located: {is_located}')
        self.last_bbox = self.state # saving last bbox
        # self.logger.info(f'last_bbox: {self.last_bbox}')

        x1, y1, w, h = self.state
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # BEN
        if not self.flip:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(255, 0 , 0), thickness=2)
        else:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0 , 255), thickness=2)

        cv2.imwrite('image_exps/frame_%s.jpg' % str(self.frame_counter).zfill(5), image)
        # self.logger.info(f'returning {BoundingBox(x1, y1, w, h, self.system_score.get_float_score())}')
        # self.logger.debug('track DONE')
        return BoundingBox(x1, y1, w, h, self.system_score.get_float_score())

    # endregion

    # region images buffer Stack functions
    def get_img_by_frame_id(self, frame_id):
        """
        Get next image from shared memory - buffer
        :return: image
        """
        #frame = cv2.imread(r'C:\Users\benew\Desktop\work\tracker_poc_deploy\bus.jpg') # BEN DELETE IT
        # self.logger.debug('get_img_by_frame_id')
        import time

        # DELETE IT AFTER INTEGRATION
        #############################
        #return self.get_last_img() ########################################
        ##########################
        # END

        counter = 1
        ret, frame = self.cap.read()
        x1, y1, w, h = self.state
        # self.logger.info(f'x1, y1, w, h: {x1, y1, w, h}')
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 255, 0), thickness=2)
        cv2.imwrite("first_file_FIRST_IMG.jpg", frame)
        # self.logger.debug('get_img_by_frame_id DONE')

        # exit(1)

        # region exmaple of video
        # frame_width = int(self.cap.get(3))
        # frame_height = int(self.cap.get(4))
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        # out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
        # while (True):
        #     ret, frame = self.cap.read()
        #     if ret == True:
        #         # Write the frame into the file 'output.avi'
        #         out.write(frame)
        #         # Display the resulting frame
        #         cv2.imshow('frame', frame)
        #         # Press Q on keyboard to stop recording
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #     # Break the loop
        #     else:
        #         break
        #         # When everything done, release the video capture and video write objects
        # self.cap.release()
        # out.release()
        # # Closes all the frames
        # cv2.destroyAllWindows()
        # endregion

        # while ret:
        #     ret, frame = self.cap.read()
        #     cv2.imshow('first_frame_%s' % counter, frame)
        #     counter += 1

        return frame

    def get_last_img(self):
        """
                Get LAST
                 image from shared memory - buffer
                :return: image
                """
        # self.logger.debug(f'get_last_img')
        #frame = cv2.imread(r'C:\Users\benew\Desktop\work\tracker_poc_deploy\bus.jpg')  # BEN DELETE IT

        # config camera might be in here or while we will run the docker container:
        # if self.cam is None:
        #     self.cam = cv2.VideoCapture(self.cam_listener)
        #     # We need to check if camera
        #     # is opened previously or not
        #     if not self.cap.isOpened():
        #         print("Error reading video file")
        #         assert self.cap.isOpened(), "Error open camera"
        #     fourcc_cap = cv2.VideoWriter_fourcc(*'MJPG')
        #     self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_cap)
        #     video_FourCC = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        #     print(video_FourCC, fourcc_cap)
        #     print("Frame default resolution:",
        #           self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'x', self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        #           "with FPS:", self.cap.get(cv2.CAP_PROP_FPS))
        #     self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        #     self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        #     self.cap.set(cv2.CAP_PROP_FPS, 30)
        #     print("Frame resolution set to:",
        #           self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'x', self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        #           "with FPS:", self.cap.get(cv2.CAP_PROP_FPS))
        #
        #     frame_width = int(self.cap.get(3))
        #     frame_height = int(self.cap.get(4))
        #
        #     # %%
        #     show_factor = min(720 / frame_height, 1)
        #     show_size = (int(show_factor * frame_width), int(show_factor * frame_height))
        #     print("show_size:", show_size)
        # ret, frame = self.cam.read()
        self.cam_frame_counter += 1
        # self.logger.info(f'cam_frame_counter: {self.cam_frame_counter}')
        # if not ret:
        #     print("failed to grab frame")
        #     return
        ret, frame = self.cap.read() # BEN READ FROM FILE
        # self.logger.info(f'frame: {frame.shape}')
        # self.logger.debug(f'get_last_img DONE')
        return frame, self.cam_frame_counter
    # endregion

    # region Auxiliary functions
    def init_hyper_params(self, path="params.yaml"):
        # self.logger.debug(f'Init hyper params')
        ym = init_params(path, 0)
        # self.seq_length = 40
        # self.add_img_to_template_score_thresh = 0.4
        # self.min_steps_to_update_template = {'attack': 2, 'hover': [2, 8]}
        # self.kalman_defualt_score = 0.41
        # self.max_acq_steps = 4
        # self.max_search_steps = 30
        # self.tracking_mode_thresh = 0.45
        # self.is_kalman_box = False
        # self.prev_is_kalman_box = False
        # self.strong_track_thresh = 0.6
        # self.ema_alpha = 0.9
        # self.update_after_hid = 15
        # self.last_hid = -self.update_after_hid
        # self.is_hid = False
        self.seq_length = ym.seq_length
        self.add_img_to_template_score_thresh = ym.add_img_to_template_score_thresh
        self.min_steps_to_update_template = ym.min_steps_to_update_template
        self.kalman_defualt_score = ym.kalman_defualt_score
        self.max_acq_steps = ym.max_acq_steps
        self.max_search_steps = ym.max_search_steps
        self.tracking_mode_thresh = ym.tracking_mode_thresh
        self.is_kalman_box = ym.is_kalman_box
        self.prev_is_kalman_box = ym.prev_is_kalman_box
        self.strong_track_thresh = ym.strong_track_thresh
        self.ema_alpha = ym.ema_alpha
        self.update_after_hid = ym.update_after_hid
        self.last_hid = ym.last_hid
        self.is_hid = ym.is_hid
        self.tracking_score_size_ratio = ym.tracking_score_size_ratio
        self.tracking_score_iou_thresh = ym.tracking_score_iou_thresh
        # self.logger.info(f'{self.seq_length}, {self.add_img_to_template_score_thresh}, {self.min_steps_to_update_template}, '
        #                 f'{self.kalman_defualt_score}, {self.max_acq_steps}, {self.max_search_steps}, '
        #                 f'{self.tracking_mode_thresh}, {self.is_kalman_box}, {self.prev_is_kalman_box}, '
        #                 f'{self.strong_track_thresh}, {self.ema_alpha}, {self.update_after_hid}, {self.last_hid}, '
        #                 f'{self.is_hid}, {self.tracking_score_size_ratio}, {self.tracking_score_iou_thresh}')

    def forward_template(self, image, bbox_by_id):
        # self.logger.debug(f'forward_template')
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, bbox_by_id, self.params.TEMPLATE_FACTOR,
                                                                output_sz=self.params.TEMPLATE_SIZE)
        # self.logger.info(f'z_patch_arr: {z_patch_arr}')
        # self.logger.info(f'z_amask_arr: {z_amask_arr}')
        self.template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        # self.logger.info(f'template: {self.template}')
        self.box_mask_z = None
        if self.params.MODEL_BACKBONE_CE_LOC:
            self.box_mask_z = generate_mask_cond(self.params, 1)
        # self.logger.info(f'box_mask_z: {self.box_mask_z}')
        # self.logger.debug('forward_template DONE')
        return

    def add_img_to_template_array(self, image, score, score_thresh=0):
        # self.logger.debug(f'add_img_to_template_array')
        if score < score_thresh:
            return
        new_id = (self.old_image['id'] + 1) % self.seq_length

        self.old_image['image'][new_id] = image
        self.old_image['bbox'][new_id] = self.state
        self.old_image['score'][new_id] = score
        self.old_image['id'] = new_id
        # self.logger.info(f'old_image: {self.old_image}')
        # self.logger.debug('add_img_to_template_array DONE')
        # if score>0.8 or (score>self.strong_track_thresh and self.frac<0.1 and self.frac>-0.1):
        #     self.old_image['init_gt'] = [image,self.state]
        #     self.best_temp_frame = self.frame_counter

    def check_sma_ok(self):
        # self.logger.debug('check_sma_ok')
        self.sma = np.average(np.array(self.old_image['bbox'])[:, 2]*np.array(self.old_image['bbox'])[:, 3])
        # self.logger.info(f'sma: {self.sma}')
        self.frac = (self.state[2]*self.state[3] - self.sma)/self.sma
        # self.logger.info(f'frac: {self.frac}')
        # self.logger.debug('check_sma_ok DONE')

    def update_sys_state(self, score, final_box):
        # self.logger.debug(f'update_sys_state')
        cur_sys_state = self.system_state.get_state()
        # self.logger.info(f'cur_sys_state: {cur_sys_state}')
        #print(str(cur_sys_state))
        if cur_sys_state == 'track':

            # self.logger.warning(f'self.is_tracking_mode(score): {self.is_tracking_mode(score)}')
            # self.logger.warning(f'is_correct_ave(score, self.last_bbox, final_box): {is_correct_ave(score, self.last_bbox, final_box)}')
            if self.is_tracking_mode(score) or is_correct_ave(score, self.last_bbox, final_box):
                #print(str(self.is_tracking_mode(score)))
                self.system_state.set_state('track')
            else:
                self.system_state.set_state('acq')

        elif cur_sys_state == 'acq':

            if self.acq_to_track(score):
                 self.system_state.set_state('track')
            elif self.is_acq_mode():
                    self.system_state.set_state('acq')
            else:
                self.system_state.set_state('search')

        elif cur_sys_state == 'search':

            if self.search_to_acq(score, final_box):
                self.system_state.set_state('acq')
            else:
                self.system_state.set_state('search')
        # self.logger.info(f'system_state: {self.system_state.get_state()}')
        # self.logger.debug('update_sys_state DONE')

    def update_sys_score(self, score):
        # self.logger.debug(f'update_sys_score')
        cur_sys_state = self.system_state.get_state()
        # self.logger.info(f'cur_sys_state: {cur_sys_state}')
        if cur_sys_state == 'track':
            self.system_score.set_score('strong_track')
        elif (self.is_stable or self.is_init_obj_shape) and score > self.tracking_mode_thresh:
            if self.system_score.get_score() == 'lost':
                if score > self.strong_track_thresh:
                    self.system_score.set_score('weak_track')
            else:
                self.system_score.set_score('weak_track')
        elif self.system_score.get_search_count() <= 3:
            self.system_score.set_score('search')
        else:
            self.system_score.set_score('lost')
        # self.logger.info(f'system_score: {self.system_score.get_score()}')
        # self.logger.debug('update_sys_score DONE')

    def is_hiding(self):
        # self.logger.debug(f'is_hiding')
        last_bbox = self.last_bbox
        # self.logger.info(f'last_bbox: {last_bbox}')
        self.is_hid = False
        front_corner, back_corner = get_front_back(last_bbox, self.state)
        # self.logger.info(f'front_corner: {front_corner}')
        # self.logger.info(f'back_corner: {back_corner}')
        front_v, back_v, center_v = get_corner_speed(last_bbox, self.state, front_corner), get_corner_speed(last_bbox, self.state, back_corner), get_corner_speed(last_bbox, self.state, 'c')
        # self.logger.info(f'front_v: {front_v}')
        # self.logger.info(f'back_v: {back_v}')
        # self.logger.info(f'center_v: {center_v}')
        treth = 1
        cond1 = front_v < center_v - treth and back_v > center_v + treth
        cond2 = self.frac < -0.3
        # self.logger.info(f'cond1: {cond1}')
        # self.logger.info(f'cond2: {cond2}')
        if cond1 and cond2:
            # self.logger.info(f'cond1 and cond2')
            self.last_hid = self.frame_counter
            self.is_hid = True
            self.KLtrack.mean[6], self.KLtrack.mean[7] = 0, 0

        # self.logger.debug(f'is_hiding DONE')

    def acq_to_track(self, score):
        t_count = self.system_state.get_time_count()
        is_track = self.is_tracking_mode(score)
        return (is_track and t_count <= self.max_acq_steps) or (score > self.tracking_mode_thresh and self.frac>-0.1 and self.frac<0.1)

    def is_tracking_mode(self, score):
        self.check_if_stable()
        self.check_obj_shape()
        return (score >= self.tracking_mode_thresh and self.is_stable and self.is_init_obj_shape) or score>self.strong_track_thresh

    def check_if_stable(self):
        x, y, w, h = self.state
        cx, cy = x + w * 0.5, y + h * 0.5
        xp, yp, wp, hp = self.last_bbox  # ASK SHAI cuz we need to update previous output
        cxp, cyp = xp + wp * 0.5, yp + hp * 0.5

        # size stable
        if w < wp - wp * self.tracking_score_size_ratio or w > wp + wp * self.tracking_score_size_ratio or\
                h > hp + hp * self.tracking_score_size_ratio or h < hp - hp * self.tracking_score_size_ratio:
            self.is_stable = False
            return

        # location stable
        if ((cx - cxp) ** 2 + (cy - cyp) ** 2) ** 0.5 > max(wp, hp):
            self.is_stable = False
            return
        #print('last box' + str(self.last_bbox))
        #print('state' +  str(self.state))
        iou_score = calc_iou_overlap(torch.tensor(np.array(self.state)).reshape(1, -1),
                                     torch.tensor(np.array(self.last_bbox)).reshape(1, -1))
                                    # torch.tensor(np.array([self.last_bbox])).reshape(1, -1)).item()
        if iou_score <= self.tracking_score_iou_thresh:
            self.is_stable = False
            return

        self.is_stable = True

    def search_to_acq(self, score, final_box):
        self.check_obj_shape()
        goto_acq = (self.is_acq_mode() and self.is_init_obj_shape) or ((self.is_stable or self.is_init_obj_shape) and score > self.strong_track_thresh)
        if self.drone_mode == 'hover':
            frac = (final_box[2]*final_box[3] - self.sma)/self.sma
            frac_ok = (frac > -0.15) and (frac < 0.15)
            goto_acq = goto_acq or frac_ok
        return goto_acq

    def is_acq_mode(self):
        if not self.prev_is_kalman_box:
            x, y, w, h = self.state
            cx, cy = x + w * 0.5, y + h * 0.5
            xp, yp, wp, hp = self.last_bbox

            cond1 = w <= wp or h <= hp  # size smaller
            cond2 = cx >= xp and cx <= xp + wp and cy >= yp and cy <= yp + hp  # center point inside previous bbox
            return cond1 and cond2
        else:
            return True

    def check_obj_shape(self):
        x,y,w,h = self.state
        xt,yt,wt,ht = self.template_update_info[-1]['bbox']
        steps_from_last_update = self.template_update_info[-1]['steps']
        ratio = 0.2 # max size change per step
        max_change = min((1+ratio) ** steps_from_last_update, 4)
        min_change = max((1-ratio) ** steps_from_last_update, 0.25)
        if w <= max_change * wt and w >= min_change * wt and h <= max_change * ht and h >= min_change * ht:
            self.is_init_obj_shape = True
            return

        self.is_init_obj_shape = False

    def check_and_update_template(self, thresh, steps_to_update, score):
        # self.logger.debug(f'check_and_update_template')
        conf_ok = torch.mean(self.old_image['score']) + thresh < self.score_avg
        # self.logger.info(f'conf_ok: {conf_ok}')
        steps_ok = self.template_update_info[-1]['steps'] > steps_to_update
        # self.logger.info(f'steps_ok: {steps_ok}')

        if (conf_ok and steps_ok) or score < 0:
            sc_mean = torch.mean(self.old_image['score'])
            # self.logger.info(f'sc_mean: {sc_mean}')

            update_id = self.old_image['id']
            # self.logger.info(f'update_id: {update_id}')

            if self.drone_mode == 'hover':
                if self.frame_counter - self.last_hid < self.update_after_hid:
                    # self.logger.info('self.frame_counter - self.last_hid < self.update_after_hid')
                    return
                if self.frac > 0.5 and not self.is_kalman_box:
                    # self.logger.info('self.frac > 0.5 and not self.is_kalman_box')
                    update_id = (self.old_image['id'] - 2) % self.seq_length
                    # self.logger.info(f'update_id: {update_id}')

            self.update_template(self.old_image['image'][update_id], self.old_image['bbox'][update_id]) # image_by_id, bbox_by_id

            self.score_avg = sc_mean if sc_mean > 0.64 else 1.
            # self.logger.info(f'score_avg: {self.score_avg}')
            #print(f' mean: {sc_mean.item()}, last: {self.score_avg}')
        # self.logger.debug(f'check_and_update_template DONE')

    def update_template(self, image_by_id,  bbox_by_id):
        # self.logger.debug(f'update_template')
        # forward the template once
        self.forward_template(image_by_id, bbox_by_id)  # only in update_teamplate we will get bbox_by_id

        # appending save states
        self.template_update_info.append({
                                'frame_counter': self.frame_counter,
                                'bbox': bbox_by_id.copy(),
                                'steps': 0
                                })
        # self.logger.info(f'template_update_info: {self.template_update_info}')
        # self.logger.debug(f'update_template DONE')
        # if self.debug:
        #     x1, y1, w, h = bbox
        #     image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
        #     save_dir = self.save_dir + '/template/' + info['seq_name'].split(' ')[0]
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_path = save_dir + ('/%04d.jpg' % self.frame_counter)
        #     cv2.imwrite(save_path, image_BGR)

    def get_final_bbox(self, kalman_bbox, score_map_ctr, size_map, offset_map, return_score=False,
                       k=5, crop2img=1, shape=(1280, 780)):
        '''
        choose k top boxes, filter the boxes over the filter threshold(set score to 0), calc the box in the real img scale and return the top box by confidence or iou
        '''
        # self.logger.debug(f'get_final_bbox')
        # self.logger.info(f'Input params: kalman_bbox: {kalman_bbox}, size_map: {size_map}, return_score: {return_score}, {k}, {crop2img}, shape: {shape}')
        # self.logger.info(f'self.sma: {self.sma}')
        # self.logger.info(f'self.feat_sz: {self.feat_sz}')
        # self.logger.info(f'self.state: {self.state}')
        # self.logger.info(f'self.drone_mode: {self.drone_mode}')
        # # self.logger.info(f'score_map_ctr: {score_map_ctr}, offset_map: {offset_map}')
        # self.logger.info(f'self.kalman_defualt_score: {self.kalman_defualt_score}')
        self.is_kalman_box = False

        def sma_filter(boxes_tensor, max_frac=1):
            frac_tensor = (boxes_tensor[:, 2] * boxes_tensor[:, 3] - self.sma) / self.sma
            filtered_idexes = (frac_tensor < max_frac).nonzero().squeeze()
            filtered_tensor = boxes_tensor[filtered_idexes]
            try:
                filtered_tensor.size(dim=1)
            except:
                filtered_tensor = filtered_tensor.unsqueeze(dim=0)
                filtered_idexes = filtered_idexes.unsqueeze(dim=0)
            return filtered_tensor, filtered_idexes

        H, W, _ = shape
        pred_boxes, pred_scores = cal_top_k_bbox(self.feat_sz, score_map_ctr, size_map, offset_map,
                                                                      return_score=return_score, k=k)
        final_boxes = clip_box_batch(map_box_back_batch(pred_boxes * crop2img, crop2img, self.state), H, W, margin=10)[0]

        if self.drone_mode == 'hover':
            max_frac = 1
            kalman_frac = (kalman_bbox[2] * kalman_bbox[3] - self.sma) / self.sma
            if kalman_frac > 0.7:
                max_frac = kalman_frac * 1.5
            filtered_bboxes, filtered_idexes = sma_filter(final_boxes, max_frac)
            final_boxes = filtered_bboxes
            pred_scores = pred_scores.squeeze()[filtered_idexes].unsqueeze(0)

        final_boxes_xyxy = ops.box_convert(final_boxes, 'xywh', 'xyxy')
        kalman_tensor = torch.tensor(kalman_bbox, device=final_boxes.device).unsqueeze(0)
        kalman_xyxy = ops.box_convert(kalman_tensor, 'xywh', 'xyxy')
        kal_iou = ops.box_iou(kalman_xyxy, final_boxes_xyxy)

        def choose_cand(method):
            if method == 'by_score':
                best_cand = torch.argmax(pred_scores[0][nms_bboxs])
            elif method == 'by_Ciou':
                best_cand = torch.argmax(Ciou_kalman_bbox)
            else:
                best_cand = 'error'
            return best_cand

        if self.drone_mode == 'hover':
            nms_bboxs = ops.nms(final_boxes_xyxy, pred_scores[0], 0.3)
            if len(nms_bboxs) > 1:
                Ciou_kalman_bbox = ops.complete_box_iou(final_boxes_xyxy[nms_bboxs], kalman_xyxy)

                Ciou_thresh = 0.2
                topk = torch.topk(Ciou_kalman_bbox, 2, dim=0, sorted=True)
                best_Ciou, scond_Ciou = topk.values[0], topk.values[1]
                if best_Ciou > scond_Ciou + Ciou_thresh:
                    method = 'by_Ciou'
                else:
                    method = 'by_score'

                best_cand = choose_cand(method=method)

                cands = [nms_bboxs, best_cand, final_boxes]
                final_box_inx = nms_bboxs[best_cand]
                final_box = final_boxes[final_box_inx]

                return final_box, torch.max(pred_scores).item(), cands

        cands = [0]

        check_kal = torch.sum(kal_iou) > 0
        search_bbox = get_search_area(self.state, self.params.SEARCH_FACTOR)
        add_kalman_cond = (check_kal and not is_contained(kalman_bbox, search_bbox) and torch.argmax(
            pred_scores) < 0.35) or pred_scores[0].size() == torch.Size([0])
        if add_kalman_cond:
            pred_scores = torch.cat(
                (pred_scores, torch.tensor([self.kalman_defualt_score], device=pred_scores.device).unsqueeze(0)), 1)
            final_boxes = torch.cat((final_boxes, kalman_tensor))

        pred_boxes = pred_boxes.view(-1, k, 4)

        max_score_idx = torch.argmax(pred_scores[0])
        if max_score_idx == len(pred_scores[0]) - 1 and add_kalman_cond:
            self.is_kalman_box = True
            self.system_state.set_state('acq')

        final_box = final_boxes[torch.argmax(pred_scores)]
        return final_box, torch.max(pred_scores).item(), cands

    # endregion
