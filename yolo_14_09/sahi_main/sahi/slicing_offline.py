# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

# import os
# import time
import sys
from pathlib import Path
# import multiprocessing
# from pathlib import Path
from typing import Dict, List, Optional, Union
import concurrent.futures
import cv2
import numpy as np
import os

# from PIL import Image
# from tqdm import tqdm
from sahi_main.sahi.utils.file import create_dir, load_json, save_json


from utils.datasets import letterbox

global image_array
global img_name
global saving_path

def create_slices(slice_height:int,
                  slice_width:int,
                  image_height: int,
                  image_width: int,
                  overlap_height_ratio=0.2,
                  overlap_width_ratio=0.2,):

    num_x_slices = np.ceil(image_width*(1+overlap_width_ratio) / slice_width)
    num_y_slices = np.ceil(image_height*(1+overlap_height_ratio) / slice_height)

    # x_overlap_pct = (num_x_slices*slice_size)/image_width - 1
    # y_overlap_pct = (num_y_slices*slice_size)/image_height - 1

    x_step = int((image_width - slice_width)/(num_x_slices-1))
    y_step = int((image_height - slice_height) / (num_y_slices - 1))

    bboxes = []

    x = 0
    while x+slice_width <= image_width:
        y = 0
        while y+slice_height <= image_height:
            bboxes.append((x, y, x+slice_width, y+slice_height))
            y = y + y_step
        x = x + x_step

    return bboxes


def slice_image_mulriproc(
    image,
    output_file_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    # out_ext: Optional[str] = None,
):
    global image_array
    global img_name
    global saving_path

    saving_path = output_dir
    from pathlib import Path

    img_name = Path(image).stem
    if output_dir:
        create_dir(output_dir)


    image_array = cv2.imread(image)
    image_width, image_height = image_array.shape[0], image_array.shape[1]
    slices_bboxes = create_slices(slice_height = slice_height,
                                  slice_width = slice_width,
                                  image_height=image_height,
                                  image_width=image_width,
                                  overlap_height_ratio=overlap_height_ratio,
                                  overlap_width_ratio=overlap_width_ratio
                                  )

    img_slices = np.zeros((len(slices_bboxes), 3, slice_height, slice_width))
    starting_pixels = []

    slices = []
    for j, bs in enumerate(slices_bboxes):
        tmp = list(bs)
        tmp.append(j)
        slices.append(tmp)

    slices_bboxes = slices.copy()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slices_bboxes)) as executor:
        for idx, i in enumerate(executor.map(crop_image, slices_bboxes)):
            img_slices[idx] = i
            starting_pixels.append([slices_bboxes[idx][0], slices_bboxes[idx][1]])

    # with multiprocessing.Pool(7) as pool:
    #     for idx, i in enumerate(pool.map(crop_image, slices_bboxes)):
    #         img_slices[idx] = i
    #         starting_pixels.append([slices_bboxes[idx][0], slices_bboxes[idx][1]])

    return img_slices, starting_pixels


def crop_image(box):
    # todo : why detections are off????
    # extract image
    img_index = box[4]
    sliced_array = image_array[box[0]: box[2], box[1]: box[3]]
    cv2.imwrite(f'{saving_path}images/{img_name}_{img_index}.jpg', sliced_array)
    img = cv2.cvtColor(sliced_array, cv2.COLOR_BGR2RGB)
    # img = letterbox(sliced_array, sliced_array.shape[1], stride=32)[0]
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img
