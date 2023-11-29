from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
from pathlib import Path

# cut_time = 2147
# end_cut_time = 2250
cut_num = 'no_black_frames'
path = "C:\\ATR_ROTEM\\IR_MOVIE\\"
video_name = "18-04-23_045233_CH0.avi"
clip = VideoFileClip(path+video_name)
clip = clip.subclip((0,299 ),(307,750),(762,900),(909,1500),(1510,1651),(1654,1801),(1806,1954),(1957,3004),(3015,4346),(4349,4361),(4371,4389) ) # NOT WORKING LIKE THIS. only 1 params.
clip.write_videofile(path + "after_"+str(cut_num)+"_" + video_name)

#ffmpeg_extract_subclip(path+video_name, cut_time, end_cut_time, path + "after_"+ str(cut_num)+ "_" + video_name)



# save video as frames.
# import cv2
# vidcap =  cv2.VideoCapture(path+video_name)
# suc,img = vidcap.read()
# count = 0
# while suc:
#     cv2.imwrite(path+'frame_%d.jpg'%count,img)
#     suc,img=vidcap.read()
#     print('Read a new frarme: ',count)
#     count += 1
