from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
#names: [ 'Person', 'on board a vessel', 'Swimmer', 'Sail boat', 'Floating object',  'Dvora', 'Zeara', 'PWC', 'Merchant Ship', 'Inflatable Boat', 'Vessel']
# from_arr = [ 0, 8 , 37] #  0->3 , 8->11 , 37 -> 8
# to_arr = [3, 11, 8]
dict_map  = { "0": "3", "8": "11", "37":"8"}

#files = glob(r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\sliced_1280_720_ol_02_no_bird_no_light_drone\labels\*\*.txt')
files = glob(r'/home/user1/ariel/improve_drone_model/data/200_imgs/labels/*.txt')
#files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\runs\detect\ben_allinone2_10_08_for_tagging_5l6\labels\*')
# path_to_write = r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\runs\detect\ben_allinone2_10_08_for_tagging_5l6\new_labels'
# Path(path_to_write).mkdir(parents=True, exist_ok=True)
counter = 1
for file in tqdm(files, total=len(files)):
    new_lines = ""
    flag = False
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        values_in_line = line.split(' ')
        if values_in_line[0] == '5':
            values_in_line[0] = '0'
            flag = True
        new_lines += ' '.join(values_in_line)
    if flag:
        with Path(file).open('w', encoding="utf-8") as f:
            counter += 1
            f.write(new_lines)
print(counter)
    # for line in lines:
    #         values_in_line = line.split(' ')
    #         values_in_line[0] = dict_map[values_in_line[0]]
    #         new_lines += ' '.join(values_in_line)
    #     with (Path(path_to_write) / Path(file).name).open('w', encoding="utf-8") as f:
    #         f.write(new_lines)
