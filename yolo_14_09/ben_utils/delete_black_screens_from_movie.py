import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from matplotlib import Path

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
count = 0
movies = glob(r'C:\ATR_ROTEM\test_movies\*.avi')
path = "C:\\ATR_ROTEM\\test_movies\\"
#video_name = "18-04-23_045233_CH0.avi"


def black_screen(img, frame_id, movie_name):
    sum = np.sum(img)
    if sum > 5000000:  # threshold 5M sum of not black pixels.
        if sum < 90000000:  # suspected black frames
            print('img sum= %s, frame_id = %s' % (sum, frame_id))
            cv2.imwrite(path + movie_name + '_' + 'frame_%d.jpg' % frame_id, img)
        return False
    return True


for movie in tqdm(movies):
    video_name = Path(movie).name
    movie_name = Path(movie).stem
    video_output_name = path + "clean_frames_" + video_name
    vidcap = cv2.VideoCapture(path+video_name)
    suc,frame = vidcap.read()
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_output_name, fourcc, fps, (width, height))
    while suc:
        suc, img = vidcap.read()
        if suc and not black_screen(img, count, movie_name):
            video.write(img)
        count += 1
    cv2.destroyAllWindows()
    video.release()










