#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json


# Truncates numbers to N decimals
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def coco_bbox_to_yolo_bbox(coco_bbox, width, height):
    dw = 1. / width
    dh = 1. / height

    xmin = coco_bbox[0]
    ymin = coco_bbox[1]
    xmax = coco_bbox[2] + coco_bbox[0]
    ymax = coco_bbox[3] + coco_bbox[1]

    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2

    w = xmax - xmin
    h = ymax - ymin

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    # Note: This assumes a single-category dataset, and thus the "0" at the beginning of each line.
    return str(
        "8 " + str(truncate(x, 7)) + " " + str(truncate(y, 7)) + " " + str(truncate(w, 7)) + " " + str(
            truncate(h, 7)))


path = r'L:\video_frames_merchant'

for folder in [os.path.join(path, folder) for folder in os.listdir(path)]:
    # read coco_json
    with open(os.path.join(folder, 'coco_json.json'), 'r') as f:
        coco_json = json.load(f)
    # read result_json
    with open(os.path.join(folder, 'result.json'), 'r') as g:
        result_json = json.load(g)

    # create a dictionary that maps image id to its filename, width, height
    id_to_hwn = {img['id']: {'height': img['height'], 'width': img['width'], 'filename': img['file_name']} for img in
                 coco_json['images']}

    # create a dictionary that maps image id to its bounding box
    id_to_bb_dict = {result['image_id']: [] for result in result_json[1:]}
    for result in result_json[1:]:
        id_to_bb_dict[result['image_id']].append(result['bbox'])
    if not os.path.exists(folder + r'/images'):
        os.mkdir(folder + r'/images')
    if not os.path.exists(folder + r'/labels'):
        os.mkdir(folder + r'/labels')
    for img_id, bbox_list in id_to_bb_dict.items():
        if img_id in id_to_hwn.keys():
            hwn = id_to_hwn[img_id]
            filename = hwn['filename']
            width = hwn['width']
            height = hwn['height']
            yolo_strings = [coco_bbox_to_yolo_bbox(bb, width, height)  for bb in bbox_list]
            full_path = os.path.join(folder, r'labels', filename[:-3] + 'txt')
            # create label file in labels folder
            with open(full_path, 'w') as f:
                f.write('\n'.join(yolo_strings))
            # move image to images folder
            if os.path.exists(folder + r'/' + filename):
                os.rename(folder + r'\\' + filename, folder + r'\\images\\' + filename)
            else:
                print(f'img {filename} does not exist')
        else:
            print(f'id {img_id} not found')

