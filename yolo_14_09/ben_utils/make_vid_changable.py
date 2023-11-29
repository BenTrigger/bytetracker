import cv2
import glob
import os
from tqdm import tqdm

#images_path = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\Erez_full_data_for_detection\Scenario1\*'
#output_path = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\Erez_full_data_for_detection\Scenario1.mp4'
images_path = r'/MyHomeDir/yolo_14_09/data/atr_zafrir/Erez_full_data_for_detection/black_panther_04-06-2021_18_37-19_12/*'
output_path = r'/MyHomeDir/yolo_14_09/data/atr_zafrir/Erez_full_data_for_detection/black_panther_04-06-2021_18_37-19_12.mp4'
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'G')
image_names = glob.glob(images_path)

first_im_path = image_names[0]
print(len(image_names))
im = cv2.imread(os.path.join(images_path, first_im_path))
frame_h, frame_w, _ = im.shape

curr_video = 0

vid = cv2.VideoWriter(output_path, fourcc, 15, (frame_w, frame_h))

for image_name in tqdm(image_names, total=len(image_names)):
    im = cv2.imread(image_name)
    vid.write(im)
vid.release()

