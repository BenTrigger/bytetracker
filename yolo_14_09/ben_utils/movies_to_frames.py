import cv2
from pathlib import Path
import imageio
from dask import delayed
import dask.array as da
import numpy as np
path = r'\\mbt.iai\dfs\AI_Group$\Data\ATR_ROTEM\RotemRecordings\10_3_22'
movie_file = Path('video_7_3_2022_13_31.avi')
p =str(Path(path,movie_file))

vidcap = imageio.get_reader(p,'ffmpeg')
shape = vidcap.get_meta_data()['size'][::-1]+(3,)
lazy_imread = delayed(vidcap.get_data)
movie = da.stack([da.from_delayed(lazy_imread(i), shape=shape, dtype=np.uint8) for i in range(vidcap.count_frames())])
count = 0
dir = Path(path, movie_file.stem)
for frame in movie:
    file_name = "_frame%d.jpg" % count
    file_name = movie_file.stem + file_name
    imageio.imwrite(str(Path(dir,file_name)), frame)
    count += 1
# vidcap = cv2.VideoCapture(p)
# success, image = vidcap.read()
# count = 0
# while success:
#     dir = Path(path, movie_file.stem)
#     file_name = "frame%d.jpg" % count
#     cv2.imwrite(str(dir) + file_name, image)
#     success, image = vidcap.read()
#     count += 1
