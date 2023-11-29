from pathlib import Path
from glob import glob
from tqdm import tqdm
import json
dict_map = {}

#path_to_write = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\runs\val\detect_val_20_10_one_pic17\names_of_files.txt' # output for predict
path_to_write = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\runs\val\detect_val_20_10_one_pic17\source_names_of_files.txt' # output source

#files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\runs\val\detect_val_20_10_one_pic17\labels\*') #input predict
files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_13_10_21\labels\val\*') #input real lbls

images = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_13_10_21\images\val\*') #input images


new_lines = []

for file in tqdm(files, total=len(files)):
    flag_in_images = False
    for img in images:
        if  Path(file).stem == Path(img).stem:
            flag_in_images = True
            break
    if flag_in_images:
        with open(file, encoding="utf-8") as f:
            try:
                lines = f.readlines()
            except Exception as e:
                print(Path(file).stem)
        for line in lines:
            values_in_line = line.split(' ')
            if values_in_line[0] == '1':  # only for 'Person on the vessel'
                new_lines.append(Path(file).stem)   #file name



new_lines = sorted(new_lines) #sorting names
ret_str = ""
for l in new_lines:
    ret_str += l+'\n'
with open(path_to_write, 'w', encoding="utf-8") as f:
    f.write(ret_str)
