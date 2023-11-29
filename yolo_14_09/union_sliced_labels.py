import cv2
import os
import time
from utils.general import xyxy2xywh
import argparse
from sahi_main.sahi.postprocess.combine import NMSPostprocess
from utils.plots import plot_images
from sahi_main.sahi.slicing_realtime import slice_image_mulriproc
from glob import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ben_utils.rectangle_imgages import rectangle_image

def run(
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.1,
    postprocess_class_agnostic: bool = False,
    output_path = None,
    filename = None,
    save_images_results = True,
    cat_names = None,
    img_path = None, # NEED TO READ from  dir all file in the same name and rewrite to unite label.txt
    arr_preds = None # NEED TO READ the image name according to the label file name
):
    image = cv2.imread(img_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # NO REASON TO DO IT BECAUSE WE TRAIN on BGR  ( utils/datasets.py line 659)
    nmspps = NMSPostprocess(match_threshold=postprocess_match_threshold, match_metric=postprocess_match_metric,
                            class_agnostic=postprocess_class_agnostic,)
    #print(arr_preds)
    full_object_prediction_list = nmspps(arr_preds)
    #print(full_object_prediction_list) # check and see that was a union :P
    #print(f'Done. ({time.time() - t0:.3f}s)')
    os.makedirs(f'{output_path}pred_labels/', exist_ok=True)
    if os.path.isfile(f'{output_path}pred_labels/{filename}.txt'):
        os.remove(f'{output_path}pred_labels/{filename}.txt')
    with open(f'{output_path}pred_labels/{filename}.txt', 'a') as myfile:
        for p in full_object_prediction_list:
            p['bbox'][0] = xyxy2xywh(np.array(p['bbox'][0]).reshape(1,-1)).flatten()
            mystring = str( str(int(p['category']))  + " " + str(truncate(float(p['bbox'][0][0]), 7)) + " " + str(truncate(float(p['bbox'][0][1]), 7)) + " "
                           + str(truncate(float(p['bbox'][0][2]), 7)) + " " + str(truncate(float(p['bbox'][0][3]), 7)))
            #BEN CHANGE FORMAT TO xyxy2xywh

            myfile.write(mystring)
            myfile.write("\n")
    myfile.close()
    if save_images_results:
        os.makedirs(f'{output_path}pred_imgs/', exist_ok=True)
        rectangle_image(output_path= f'{output_path}pred_imgs/{filename}.png', im=image, preds_as_bbox =full_object_prediction_list)
        #plot_images(images=image, preds=full_object_prediction_list, fname=f'{output_path}pred_labels/{filename}.jpg')


def sliced_folder(path_to_sliced_preds, original_imgs_not_sliced):
    ret = {}
    for img_path in original_imgs_not_sliced:
        img_stem = Path(img_path).stem
        preds_belong_to_img = list(filter(lambda x: img_stem + '_' in x, path_to_sliced_preds))
        # print('img full path: %s'% img_path)
        # print('lbls for that img: %s' % lbls_belong_to_img)
        ret[img_path] = preds_belong_to_img
    return ret


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def read_preds(arr_preds_path,starting_pixels, slice_size):
    ret = []
    img_w = 3840
    img_h = 2160
    for val in arr_preds_path:
        idx_of_frame = int(val[-5])
        filename = Path(val).name
        with open(val, 'r', encoding='UTF-8') as file:
            lines = file.readlines()
            for line in lines:
                x = line.rstrip().split(' ')
                #print(x)
                if len(x) < 5 : continue
                x = np.array(x).astype(np.float)
                x[1:] *= slice_size
                c_x1 = (starting_pixels[idx_of_frame][1] + x[1]) / img_w  # get x1 center  - x in index j 1
                c_y1 = (starting_pixels[idx_of_frame][0] + x[2]) / img_h  # get y1 center  - y in index i 0
                w = x[3] / img_w
                h = x[4] / img_h  # get H
                # ret.append([int(x[0]), # LABEL
                #             (c_x1 - w/2),   # X1
                #             (c_y1 - h/2),   # Y1
                #             (c_x1 + w/2),   # X2
                #             (c_y1 + h/2)])  # Y2
                # data slicer source:  [(0, 0, 1600, 1600), (0, 1120, 1600, 2720), (0, 2240, 1600, 3840), (560, 0, 2160, 1600), (560, 1120, 2160, 2720), (560, 2240, 2160, 3840)]
                # [[0, 0], [560, 0], [0, 1120], [560, 1120], [0, 2240], [560, 2240]] starting pixle after sorting
                ret.append(
                    ({'file': filename,
                            'bbox': [[(c_x1 - w/2), (c_y1 - h/2), (c_x1 + w/2), (c_y1 + h/2)],
                                [w, h]],
                            'score': 1.0, # NEED TO CHANGE TO REAL SCORE IF POSSIBLE
                            'category': float(x[0])})
                )
    return ret


if __name__ == '__main__':
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='') # MUST ENTER THE REAL TAGGED DIRECTORT: /MyHomeDir/yolo_14_09/data/atr_zafrir/tagged_14_02_22/
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1600, help='train, val image size (pixels)')
    parser.add_argument('--name', default='', help='save to project/name')
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--conf_thres', type=lambda s: [float(item) for item in s.split(',')], default=[0.001], help='confidence threshold')
    opt = parser.parse_args()

    postprocess_match_metric = "IOU"
    postprocess_match_threshold = 0.1
    postprocess_class_agnostic =  False
    output_path = r'/MyHomeDir/yolo_14_09/check_res/'
    filename = None
    save_images_results = True
    cat_names = [ 'Person', 'Person on board a vessel', 'Swimmer', 'Sail boat', 'Floating object',  'Dvora', 'Zeara', 'PWC', 'Merchant Ship', 'Inflatable Boat', 'Vessel']

    # just to know the real width height of the pictures
    original_imgs_sliced_path = opt.data + r'sliced/images/val/*'
    # to know the real starting pixels
    original_imgs_not_sliced_path = opt.data + r'images/val/*'
    # Path to sliced predicted labels.
    path_to_sliced_preds_path = r'/MyHomeDir/yolo_14_09/runs/val/val_on_val_set_sliced_14_02_222/labels/*.txt'

    path_to_sliced_preds = sorted(glob(path_to_sliced_preds_path))
    files_sliced,files_not_sliced = [], []
    files_sliced.extend(glob(original_imgs_sliced_path + '.jpg'))
    files_sliced.extend(glob(original_imgs_sliced_path + '.png'))
    original_imgs_sliced = sorted(files_sliced)
    files_not_sliced.extend(glob(original_imgs_not_sliced_path + '.jpg'))
    files_not_sliced.extend(glob(original_imgs_not_sliced_path + '.png'))
    original_imgs_not_sliced = sorted(files_not_sliced)
    im = cv2.imread(original_imgs_not_sliced[0])
    sliced_img = cv2.imread(original_imgs_sliced[0])
    sliced_img_h, sliced_img_w, sliced_channels = sliced_img.shape
    slices, starting_pixels = slice_image_mulriproc(
        image=im,
        slice_height=sliced_img_h,
        slice_width=sliced_img_w,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        output_dir=None,
        output_file_name=None
    )
    starting_pixels.sort(key=lambda x: x[1])  # DONT FORGET TO SORT LIKE THIS I=Y , J=X
    #print(starting_pixels)
    dic_img_and_preds = sliced_folder(path_to_sliced_preds, original_imgs_not_sliced)
    # for key,value in dic_img_and_preds.items():
    #     print(key)
    #     print(value)
    #     print("##############################################################################")
    # exit(1)
    print(len(dic_img_and_preds.items()))
    for img_path, arr_preds_path in tqdm(dic_img_and_preds.items(), total=len(dic_img_and_preds.items())): # run on all frames, find all labels files for taht frame and unite them
        arr_preds_for_one_img = read_preds(arr_preds_path, starting_pixels, opt.imgsz)
        # print(Path(arr_preds_path[0]).stem)
        try:
            run(postprocess_match_metric, postprocess_match_threshold, postprocess_class_agnostic, output_path, Path(arr_preds_path[0]).stem[:-2], save_images_results,
                cat_names, img_path, arr_preds_for_one_img)
        except Exception as e:
            print(e)
            print(arr_preds_path)
