import os
import glob
from shutil import copy2

src_img, scr_lbl, dst_img, dst_lbl = '/home/user1/ariel/improve_drone_model/data/Ben_img/images/','/home/user1/ariel/improve_drone_model/data/Ben_img/labels/','/home/user1/ariel/yolov5_quant_sample/b_data/orig_img/','/home/user1/ariel/yolov5_quant_sample/b_data/orig_label/'

src_img_list= glob.glob(os.path.join(src_img,'*.png'))
src_img_list.sort()
for file in src_img_list[:550]:
    copy2(file,dst_img+src_img.split('/')[-1])
    copy2(scr_lbl+file.split('/')[-1].replace('.png','.txt'), dst_lbl+src_img.split('/')[-1].replace('.png','.txt'))