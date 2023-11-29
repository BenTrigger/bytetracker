from pathlib import Path
from glob import glob
from tqdm import tqdm
import shutil

lbl_files = glob(r'Z:\yolo_14_09\data\ROTEM_NEW_2023\source_labels_part2\*\*\obj_train_data\*')
path_to_search = Path(r'Z:\yolo_14_09\data\ROTEM_NEW_2023\source_images')
path_to_write_img = Path(r'Z:\yolo_14_09\data\ROTEM_NEW_2023\images')
path_to_write_lbl = Path(r'Z:\yolo_14_09\data\ROTEM_NEW_2023\labels')

for file in tqdm(lbl_files, total=len(lbl_files)):
    project_name = Path(file).parents[2].stem
    img_found = path_to_search / project_name / (str(Path(file).stem) +'.JPG') # FIX THAT OR CHECK IT BEFORE , JPS BIG CAPITAL
    if Path(img_found).exists():
        shutil.copy(str(img_found), str(path_to_write_img / (project_name + '_' + (str(Path(file).stem) +'.jpg'))))
        shutil.copy(file, str(path_to_write_lbl / (project_name + '_' + (str(Path(file).stem) +'.txt'))))

