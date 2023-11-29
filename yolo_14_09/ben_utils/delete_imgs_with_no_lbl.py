from pathlib import Path
from glob import glob
from tqdm import tqdm
import os

files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\ROTEM_NEW_2023\\images\*.jpg')
path_to_search = Path(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\ROTEM_NEW_2023\labels')

for file in tqdm(files, total=len(files)):
    lbl_to_find = path_to_search / (str(Path(file).stem) +'.txt')
    if Path(lbl_to_find).exists():
        pass
    else:
        os.remove(file)
