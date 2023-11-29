
from matplotlib import Path
from glob import glob
import os
from tqdm import tqdm
import cv2

###
# Folder with images
###
image_folder = r'C:\ATR_ROTEM\IR_MOVIE\*'
# SORT BY LEN FIX THE PROBLEM (IF YOU CHANGE NAME FORMAT - DON'T FORGET TO CHECK IT)
sorted_images_paths = sorted(glob(image_folder), key=len)
path = Path(image_folder).parents[0]
video_output_name = str(Path(image_folder).parents[1] / (str(path.name) + '_h264.mp4'))
print(video_output_name)

Flag = True
for img in tqdm(sorted_images_paths):
    try:
        if Flag:
            frame = cv2.imread(img)
            height, width, layers = frame.shape
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc(*'h264')
            #fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            fps  = 30
            video = cv2.VideoWriter(video_output_name, fourcc, fps, (width, height))
            Flag = False
        video.write(cv2.imread(img))
    except Exception as e:
        print(e)
cv2.destroyAllWindows()
video.release()

###
    #SUB-FOLDERS
###
#image_folder = r'\\mbt.iai\dfs\AI_Group$\Data\ATR_ROTEM\RotemRecordings\9_Aug_22\*\*'
# folders = glob(image_folder)
# for images_paths in tqdm(folders):
#     if not Path(images_paths).is_dir():
#         continue
#     video_output_name = str(Path(images_paths).parents[0] / (str(Path(images_paths).name) + '.avi'))
#     Flag = True
#     sorted_images_paths = sorted(glob(images_paths + '\*.JPG'), key=len) # SORT BY LEN FIX THE PROBLEM (IF YOU CHANGE NAME FORMAT - DON'T FORGET TO CHECK IT)
#     for img in tqdm(sorted_images_paths):
#         if Flag:
#             frame = cv2.imread(img)
#             height, width, layers = frame.shape
#             print(video_output_name)
#             video = cv2.VideoWriter(video_output_name, 0, 10, (width, height))
#             Flag = False
#         video.write(cv2.imread(img))
#
#     cv2.destroyAllWindows()
#     video.release()


