import math
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## REMOVE LABELS FROM TEXT FILES ##


#files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\core_labels_for_test\labels\*\*.txt')

#files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\sliced_2160_ol_02_no_bird_no_light_drone\labels\*\*.txt')
#files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\sliced_1280_720_ol_02_no_bird_no_light_drone\labels\*\*.txt')
files = glob(r'/MyHomeDir/yolo_14_09/data/Baloons_DATA/tagged_2_2_2023_all_frames/extra_images_10_7_23_SELCTED/sliced/labels/*/*.txt')
#files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\sliced\only_core_labels\labels\*\*.txt')
#files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\sliced\only_core_labels\check_code.txt')

for file in tqdm(files, total=len(files)):
    lines_to_write = []
    flag = False
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        values_in_line = line.split(' ')
        if int(values_in_line[0]) == 2 :#or int(values_in_line[0]) == 2 or int(values_in_line[0]) == 3 \
                #or int(values_in_line[0]) == 4 or int(values_in_line[0]) == 8: # 'Bird','UFO','AirPlane_w_lights',other
            flag = True
            continue
        lines_to_write.append(line)
    if flag:
        with open(file, 'w') as fw:
            fw.writelines(lines_to_write)


## CHANGE LABELS FROM TEXT FILES ##

# files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\sliced\only_core_labels\labels\*\*.txt')
# #files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\sliced\only_core_labels\check_code.txt')
#
#
# for file in tqdm(files, total=len(files)):
#     lines_to_write = []
#     flag = False
#     with open(file) as f:
#         lines = f.readlines()
#     for line in lines:
#         values_in_line = line.split(' ')
#         if int(values_in_line[0]) == 5:
#             line = '2' + line[1:]
#         elif int(values_in_line[0]) == 6:
#             line = '3' + line[1:]
#         elif int(values_in_line[0]) == 7:
#             line = '4' + line[1:]
#         lines_to_write.append(line)
#     with open(file, 'w') as fw:
#         fw.writelines(lines_to_write)






