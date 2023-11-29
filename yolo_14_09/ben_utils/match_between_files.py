from glob import glob
from tqdm import tqdm
from matplotlib import Path
import shutil

#files_imgs = glob(r'Z:\yolo_14_09\data\atr_zafrir\CVAT_Matrix\*\obj_train_data\*.PNG')
files_txt = glob(r'Z:\yolo_14_09\data\atr_zafrir\CVAT_Matrix\*\obj_train_data\*.txt')
dest_to_write = r'Z:\yolo_14_09\data\atr_zafrir\tagged_youtube\images\train'
format_to_find = ".PNG"
for txt_path in tqdm(files_txt, total=len(files_txt)):
    src = Path(txt_path)
    project = str(src.parents[1]).split("\\")[-1]
    inside_folder = str(src.parents[0]).split("\\")[-1]
    directory = src.parents[1]
    file_stem = src.stem
    file_name = src.name
    if Path(directory,inside_folder,file_stem + format_to_find).exists():
        Path(dest_to_write,project).mkdir(exist_ok=True)
        shutil.copy(str(Path(directory,inside_folder,file_stem + format_to_find)), str(Path(dest_to_write,project, project+'_'+file_stem+format_to_find)))
        shutil.copy(str(txt_path), str(Path(dest_to_write,project, project+'_'+file_name)))

