import math
from pathlib import Path
from glob import glob
from tqdm import tqdm
import shutil
import pandas as pd
import matplotlib.pyplot as plt


files = glob(r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\extra_images_10_7_23_SELCTED\images\train\*')
path_to_write = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\extra_images_10_7_23_SELCTED\labels\train'
path_to_search = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\extra_images_10_7_23\labels\train'

for file in tqdm(files, total=len(files)):
    lbl_file = Path(path_to_search) / (str(Path(file).stem) +'.txt')
    if lbl_file.exists():
        shutil.copy(str(lbl_file), Path(path_to_write) / (str(Path(file).stem) +'.txt') )










