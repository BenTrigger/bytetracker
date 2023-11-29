from pathlib import Path
from glob import glob
from tqdm import tqdm
import json
dict_map = {}

# path_to_write = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\before_slice_test_set_count_types_source.txt'
# files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\labels\test\*.txt')
path_to_write = r'/MyHomeDir/yolo_14_09/data/Baloons_DATA/tagged_2_2_2023_all_frames/extra_images_10_7_23/after_slice_count_types_no_birds_source5.txt'
files = glob(r'/MyHomeDir/yolo_14_09/data/Baloons_DATA/retagged_7_8_22/Detection_Training_Datasets/Ibdis_21-05-23_7C/labels/*.txt')


for file in tqdm(files, total=len(files)):
    new_lines = ""
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        values_in_line = line.split(' ')
        if values_in_line[0] in dict_map.keys():
            dict_map[values_in_line[0]] += 1
        else:
            dict_map[values_in_line[0]] = 1

    with open(path_to_write, 'w', encoding="utf-8") as f:
        json.dump(dict_map, f)
