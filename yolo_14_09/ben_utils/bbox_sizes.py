import math
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

## ALL TYPES TOGETHER
path_to_write = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\bboxes_size_Train_Set.txt'
#path_to_copy_fixed_label = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\\area_above_64_labels\train\\'
files = glob(r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\labels\train\*.txt')
#dic_range = {'4':0, '9':0, '16':0, '25':0, '36':0, '49':0, '64':0,'bigger':0}  #,'81':0,'100':0,'bigger':0}

dic_range = {}
for i in range(64):
    dic_range[str(i)] = 0
dic_range["bigger"] = 0
dic_range = {'xs':0, 'small':0, 'med':0, 'big':0}
#arr_sizes = []
for file in tqdm(files, total=len(files)):
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        values_in_line = line.split(' ')
        area_in_pixels = int(float(values_in_line[3]) * 3840 * float(values_in_line[4]) * 2160) #NEED TO CHANGE TO SLICED IMAGE SIZE !!!! not 4k
        sqrt_val = int(math.sqrt(area_in_pixels))
        if sqrt_val > 64: # 64 = 4096 pixels.
            dic_range['big'] += 1
        elif sqrt_val >28:
            dic_range['med'] += 1
        elif sqrt_val >13:
            dic_range['small'] += 1
        elif sqrt_val >0:
            dic_range['xs'] += 1
            #fw.write(line)
        else:
            dic_range[str(sqrt_val)] += 1
            #arr_sizes.append(area_in_pixels)
    #fw.close()
    with open(path_to_write,'a') as f:
        f.write(str(dic_range))
### end

### FOR EACH TYPE(OBJ)
# path_to_write = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\size_by_type\bboxes_size_Train_Set_by_types.txt'
# #path_to_copy_fixed_label = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\\area_above_64_labels\train\\'
# files = glob(r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\labels\train\*.txt')
# #dic_range = {'4':0, '9':0, '16':0, '25':0, '36':0, '49':0, '64':0,'bigger':0}  #,'81':0,'100':0,'bigger':0}
# obj_type = [0,1,2,3,4,5,6,7,8]
#
# # dic_range = {}
# # for i in range(64):
# #   dic_range[str(i)] = 0
# # dic_range["bigger"] = 0
# for i in obj_type:
#     dic_range = {'xs':0, 'small':0, 'med':0, 'big':0}
#     arr_sizes = []
#     for file in tqdm(files, total=len(files)):
#         with open(file) as f:
#             lines = f.readlines()
#         #fw = open(path_to_copy_fixed_label + str(Path(file).name), 'w')
#         for line in lines:
#             values_in_line = line.split(' ')
#             if int(values_in_line[0]) == i:
#                 area_in_pixels = int(float(values_in_line[3]) * 3840 * float(values_in_line[4]) * 2160) #NEED TO CHANGE TO SLICED IMAGE SIZE !!!! not 4k
#                 sqrt_val = int(math.sqrt(area_in_pixels))
#                 if sqrt_val > 64: # 64 = 4096 pixels.
#                     dic_range['big'] += 1
#                 elif sqrt_val >28:
#                     dic_range['med'] += 1
#                 elif sqrt_val >13:
#                     dic_range['small'] += 1
#                 elif sqrt_val >0:
#                     dic_range['xs'] += 1
#                     #fw.write(line)
#                 else:
#                     dic_range[str(sqrt_val)] += 1
#                 #arr_sizes.append(area_in_pixels)
#         #fw.close()
#     with open(path_to_write,'a') as f:
#         f.write("type %s:" % i)
#         f.write(os.linesep)
#         f.write(str(dic_range))
#         f.write(os.linesep)
### end

### FOR EACH TYPE(OBJ) IN SUB-FOLDERS
# import pandas as pd
# df = pd.DataFrame(columns=['Drone','AirPlane','Bird','UFO','AirPlane_w_lights','Baloons','Drone_w_lights','Single_front_lights','other'])
# obj_type = [0,1,2,3,4,5,6,7,8]
# names = ['Drone','AirPlane','Bird','UFO','AirPlane_w_lights','Baloons','Drone_w_lights','Single_front_lights','other']
# folders = glob(r'Z:\yolo_14_09\data\Baloons_DATA\retagged_7_8_22\Detection_Training_Datasets\*')
# out_put_csv = r'Z:\yolo_14_09\data\Baloons_DATA\retagged_7_8_22\Detection_Training_Datasets\bboxes_by_types_train.csv'
# for folder in folders:
#     files = glob(str(Path(folder) / "labels\*.txt"))
#     path_to_write = str(Path(folder)) +  "_bboxes_size_by_types.txt"
#     raw_to_csv = [0,0,0,0,0,0,0,0,0]
#     for i in obj_type:
#         dic_range = {'xs':0, 'small':0, 'med':0, 'big':0}
#         arr_sizes = []
#         for file in tqdm(files, total=len(files)):
#             with open(file) as f:
#                 lines = f.readlines()
#             #fw = open(path_to_copy_fixed_label + str(Path(file).name), 'w')
#             for line in lines:
#                 values_in_line = line.split(' ')
#                 if int(values_in_line[0]) == i:
#                     area_in_pixels = int(float(values_in_line[3]) * 3840 * float(values_in_line[4]) * 2160) #NEED TO CHANGE TO SLICED IMAGE SIZE !!!! not 4k
#                     sqrt_val = int(math.sqrt(area_in_pixels))
#                     if sqrt_val > 64: # 64 = 4096 pixels.
#                         dic_range['big'] += 1
#                     elif sqrt_val >28:
#                         dic_range['med'] += 1
#                     elif sqrt_val >13:
#                         dic_range['small'] += 1
#                     elif sqrt_val >0:
#                         dic_range['xs'] += 1
#                         #fw.write(line)
#                     else:
#                         dic_range[str(sqrt_val)] += 1
#                     #arr_sizes.append(area_in_pixels)
#             #fw.close()
#         # with open(path_to_write,'a') as f:
#         #     f.write(names[i])
#         #     f.write(str(dic_range)+ "\r\n")
#         raw_to_csv[i] = str(dic_range['xs']) + ',' +  str(dic_range['small']) + ',' + str(dic_range['med']) + ',' + str(dic_range['big'])
#     df.loc[Path(folder).name] = raw_to_csv[:] #[raw_to_csv[0],raw_to_csv[1],raw_to_csv[2],raw_to_csv[3],raw_to_csv[4],raw_to_csv[5],raw_to_csv[6],raw_to_csv[7],raw_to_csv[8]]
#     #print(df)
# df.to_csv(out_put_csv, encoding='utf-8')
### end










# names = {0: 'Drone', 1: 'AirPlane', 2: 'Bird', 3: 'UFO', 4: 'Airplane_w_lights',  5: 'Baloons',
#          6: 'Drone_w_lights', 7: 'single_front_light', 8: 'other'}
#
# #arr_sizes = [300, 1000, 1005, 1010 , 4000, 7000, 12000]
# arr_sizes = np.genfromtxt(r'Z:\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\bboxes_size.txt',delimiter=',', dtype=np.int32)
# arr_sizes[0] = 468
# # bins = int(max(arr_sizes) / 10)
# plt.xlabel('varible x(bin size')
# plt.ylabel('count')
#
# plt.savefig(r'Z:\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\graph_for_bbox_sizes.png')
# plt.show()



