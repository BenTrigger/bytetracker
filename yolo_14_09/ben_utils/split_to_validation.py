from pathlib import Path
from glob import glob
from tqdm import tqdm
import shutil
import json
with open(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\count_types.json') as f:
    dict_map = json.load(f)
arr_counting = [0,0,0,0,0,0,0,0,0,0,0]
path_to_write_image_train = Path(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\outputs\images\train')
path_to_write_label_train = Path(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\outputs\labels\train')
path_to_write_image_val = Path(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\outputs\images\val')
path_to_write_label_val = Path(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\outputs\labels\val')


source_images_path = Path(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\outputs\sliced_images')

precentage_split = 0.15
sorted_by_val = {k: v for k, v in sorted(dict_map.items(), key=lambda item: item[1])}

for type in sorted_by_val.keys():
    files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\outputs\sliced_labels\*')
    type = int(type)
    for file in tqdm(files, total=len(files)):
        new_lines = ""
        with open(file) as f:
            lines = f.readlines()
        for line in lines:
            values_in_line = line.split(' ')
            if  type == int(values_in_line[0]) and arr_counting[type-1] < precentage_split * int(dict_map[str(type)]): #number of this type.
                arr_counting[type-1] += 1
                #move files right here here
                shutil.move(file, path_to_write_label_val / (str(Path(file).stem) +'.txt') )
                img_to_cpy = source_images_path / (str(Path(file).stem) +'.png')
                shutil.move(img_to_cpy, path_to_write_image_val / (str(Path(file).stem) +'.png') )
                break
        if arr_counting[type-1] >= precentage_split * int(dict_map[str(type)]):
            break


