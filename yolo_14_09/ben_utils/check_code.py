#
# from matplotlib import Path
# # import albumentations as a
# # import cv2
# epochs = 250
# for epoch in range(epochs):
#     last = r'Z:\path1\path2\img.pt'
#     last = str( Path(last).parents[0] / (Path(last).stem + '_' + str(epoch) + '.pt'))
#     print(last)
#     exit(1)

# transform = a.Compose([
#                 a.Defocus(radius=(30,30), alias_blur=(0.5, 0.5), always_apply=True)])
# img = cv2.imread(r'Z:\Img132101859620496209.png')
# #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# transformed = transform(image=img)
# transform_image = transformed["image"]
#
# cv2.imwrite(r'Z:\Img132101859620496209_blur_defocus.png', transform_image)



# import numpy as np
# from pathlib import Path
# p = 'black_panther_04-06-2021_10_10-10_14_cam_1_frame_650_3.txt'
#
# print(p[:-6])
# exit(1)
# a = np.array([2, 1, 4, 1])
# b = np.array([1, 2, 3, 4])
# # print(np.maximum(a[:2], b[:2]))
# # print("")
# # print(a[:2])
# # print(b[:2])
# print(b[1:3])
# # data = {}
# # data['haha']='haha1'
# # data['kaka']='kaka2'
# #
# # with open('kaka.json','w',encoding='utf-8') as f:
# #     json.dump(data,f,ensure_ascii=False, indent=4)

# a = [1,2,3,4,5,6]
# print(a[1:5])
# exit(1)
# id = 0
# cout_swimmers_id = {}
# for i in range(10):
#     if i % 2 == 0:
#         id = i
#     if not cout_swimmers_id:
#         cout_swimmers_id[id] = 1
#     elif id not in cout_swimmers_id.keys():
#         cout_swimmers_id[id] = 1
#     else:
#         cout_swimmers_id[id] += 1
# for key in cout_swimmers_id.keys():
#     print('swimmers id: %d counter %d' % (key,cout_swimmers_id[key] ))
# from time import time
# t0 = time()
# from datetime import datetime
# st = datetime.now()
# for i in range(100000):
#     print("x")
# end = datetime.now()
# t1 = time()
# print(t1-t0)
# # diff = end - st
# seconds_in_day = 24*60*60
# x = divmod(diff.days * seconds_in_day + diff.seconds, 60)
# print(x)



# import numpy as np
# a = np.array([[1,2,3],
#     [4,-5,6],
#     [7,8,9]])
#
# print( a[a[:,1] > 0])

#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# names = {0: 'Person', 1: 'Person on board a vessel', 2: 'Swimmer', 3: 'Sail boat', 4: 'Floating object',  5: 'Dvora',
#          6: 'Zeara', 7: 'PWC', 8: 'Merchant Ship', 9: 'Inflatable Boat', 10: 'Vessel'}
# df = pd.read_csv(r'Z:\deepsort_yolov5\deepsort_ben\runs\track\mul_4_custom_weights_all_data_deep_crop_226\object_sizes\all_together.csv')
# df['area'] = df['width'] * df['height']
# for i in range(11):
#     type_df = df[df['Label'] == i]
#     for name in ['width', 'height' , 'area']:
#         plt.xlim(0, max(type_df[name].values))
#         plt.xlabel('%s val(Pixels)' % name)
#         plt.ylabel('amount')
#         plt.title('%s graph for %s'% (name,names[i]))
#         type_df[name].hist(bins=100)
#         plt.savefig(r'Z:\deepsort_yolov5\deepsort_ben\runs\track\mul_4_custom_weights_all_data_deep_crop_226\object_sizes\graphs\%s_graph_for_%s.png' % (name, names[i]))
#         plt.figure().clear()
#         plt.close()
#         plt.cla()
#         plt.clf()
#ax = df.hist(bins=10)

#fig.savefig(r'Z:\deepsort_yolov5\deepsort_ben\runs\track\mul_4_custom_weights_all_data_deep_crop_226\object_sizes\graph.png')


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())
torch.zeros(1, 3, 3840, 3840).to('cuda')
exit(1)

#
#
# def add(matrix,count, max_det,  i, j, detections):
#     # add a single prediction of xyxy to the matrix
#     #detections = xyxy2xywh(detections)
#     for det in detections[:,:4]:
#         _,_,w,h = det.astype(int)
#         n = count[i,j]
#         matrix[i, j, n, 0] = h
#         matrix[i, j, n, 1] = w
#         count[i, j] += 1 # current possition
#         if count[i, j] == max_det - 1:
#             print(f"reach max det in Bbox sizes {count[i, j]} at i {i}, j {j}")
#     return matrix , count
#
# def print(matrix):
#     matrix[:,:,:,0] *= matrix[:,:,:,1]
#     print(np.mean(matrix[:,:,:,0], axis=2))
#
#
# def plot(matrix, count, nc,  save_dir='', names=()):
#     matrix[:,:,:,0] *= matrix[:,:,:,1]
#     array = np.sum(matrix[:,:,:,0], axis=2) / count
#     print(array)
#     array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
#
#     fig = plt.figure(figsize=(12, 9), tight_layout=True)
#     sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
#     labels = (0 < len(names) < 99) and len(names) == nc  # apply names to ticklabels
#
#     sn.heatmap(array, annot=nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
#                 xticklabels=names + ['background FP'] if labels else "auto",
#                 yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
#     fig.axes[0].set_xlabel('True')
#     fig.axes[0].set_ylabel('Predicted')
#     #fig.savefig(Path(save_dir) / f'bbox_sizes_confusion_matrix.png', dpi=250)
#     plt.close()
#
#
# nc = 11
# max_det=10000
# matrix = np.zeros((nc + 1, nc + 1, max_det, 2)) # pred class X label class X max detections num X h X w
# shape = matrix.shape
# count = np.zeros((nc + 1, nc + 1), dtype=np.int32)
# detections = np.array([[3.0, 0.0168538, 0.511111, 0.0172038, 0.05, 0.941406],
#               [10.0, 0.797917, 0.508333, 0.00520833, 0.00740741, 0.812988],
#               [10.0, 0.269792, 0.499537, 0.00208333, 0.0037037, 0.617676],
#               [10.0, 0.269792, 0.459537, 0.00308333, 0.0037037, 0.617676]])
# matrix, count = add(matrix, count, max_det,0,0, detections)
# print(matrix)
