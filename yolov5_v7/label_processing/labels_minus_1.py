from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np





#files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\labels\*')
#files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\retagged_7_8_22\no_birds_test_dataset\*\labels\*.txt')
####files = glob(r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\sliced_1600_no_bird_no_light_drone\labels\*\*.txt') # ERROR !!! NEED TO RE PROCCESSS THIS !
#files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\sliced_2160_ol_02_no_bird_no_light_drone\labels\*\*.txt')
#files = glob(r'/MyHomeDir/yolo_14_09/data/Baloons_DATA/tagged_2_2_2023_all_frames/sliced_1280_720_ol_02_no_bird_no_light_drone/labels/*/*.txt')
files = glob(r'/home/user1/ariel/improve_drone_model/data/200_imgs/labels/*.txt')

for file in tqdm(files, total=len(files)):
    new_lines = ""
    flag = False
    with open(file) as f:
        #print(file)
        lines = f.readlines()
    for line in lines:
        values_in_line = line.split(' ')
        #if int(values_in_line[0]) == 0 or int(values_in_line[0]) == 1: # FOR BIRDS
        if int(values_in_line[0]) == 0 or int(values_in_line[0]) == 1 or int(values_in_line[0]) == 2\
              or int(values_in_line[0]) == 3 or int(values_in_line[0]) == 4: # FOR Drone With Light
            pass
        #elif int(values_in_line[0]) == 2:  # FOR BIRDS
        elif int(values_in_line[0]) == 5:
        #    print("error, should not be Bird = 2 in here")
            print("error, should not be dron with light = 5 in here")
            exit(1)
        else:
            flag = True
            values_in_line[0] = str(int(values_in_line[0]) - 1)
        new_lines += ' '.join(values_in_line)
    if flag:
        with open(file, 'w') as fw:
            fw.writelines(new_lines)






# dict_map  = { "0": "3", "8": "11", "37":"8"}
#
# #path_to_write = r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\outputs\sliced_labels\new_labels_minus_1'
# path_to_write = r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\labels\new_labels_minus_1'
# files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\labels\*')
# #files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolov5-master_20_07\data\atr_zafrir\tagged_data_29_8\outputs\sliced_labels\*')
# Path(path_to_write).mkdir(parents=True, exist_ok=True)
#
# for file in tqdm(files, total=len(files)):
#     new_lines = ""
#     with open(file) as f:
#         lines = f.readlines()
#     for line in lines:
#         values_in_line = line.split(' ')
#         values_in_line[0] = str(int(values_in_line[0]) - 1)
#         new_lines += ' '.join(values_in_line)
#     with (Path(path_to_write) / Path(file).name).open('w', encoding="utf-8") as f:
#         f.write(new_lines)
