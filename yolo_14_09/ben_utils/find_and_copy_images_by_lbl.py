from pathlib import Path
from glob import glob
from tqdm import tqdm
import shutil


def from_all_in_one():
    path_to_search_1 = Path(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_rotem\NEW_MOVIES_7_3_22\video_7_3_2022_11_56')
    path_to_search_2 = Path(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_rotem\NEW_MOVIES_7_3_22\video_7_3_2022_13_31')
    lbl_files = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_rotem\tagged_data_for_train\labels\*')
    path_to_write = Path(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_rotem\tagged_data_for_train\images')

    for file in tqdm(lbl_files, total=len(lbl_files)):
        img_in_1 = path_to_search_1 / (str(Path(file).stem) +'.jpg')
        img_in_2 = path_to_search_2 / (str(Path(file).stem) +'.jpg')
        if Path(img_in_1).exists():
            shutil.copy(img_in_1, path_to_write / (str(Path(file).stem) +'.jpg') )
        elif Path(img_in_2).exists():
            shutil.copy(img_in_2, path_to_write / (str(Path(file).stem) +'.jpg') )

    # for line in lines:
    #     values_in_line = line.split(' ')
    #     values_in_line[0] = dict_map[values_in_line[0]]
    #     new_lines += ' '.join(values_in_line)
    # with (Path(path_to_write) / Path(file).name).open('w', encoding="utf-8") as f:
    #     f.write(new_lines)


def from_sub_folders(val_arr = None):
    if not val_arr:
        val_arr = ['black_panther_04-06-2021_21_23-21_25_cam_1',
               'black_panther_04-06-2021_10_10-10_14_cam_1',
               'black_panther_04-06-2021_11_55-11_58_cam_1',
               'black_panther_04-06-2021_11_55-11_58_cam_2',
               'black_panther_04-06-2021_19_20-19_25_cam_1']
    path_to_search = glob(r'\\mbt.iai\dfs\AI_Group$\Data\TMM_ATR\shore_security\results\bp_labels\*')
    for sub_folder in tqdm(path_to_search, total=len(path_to_search)):
        path_to_write_images = Path(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_13_10_21\images')
        path_to_write_lbls = Path(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_13_10_21\labels')

        if Path(sub_folder).stem in val_arr:  # change path to val
            path_to_write_images = path_to_write_images / 'val'
            path_to_write_lbls = path_to_write_lbls / 'val'
        else:
            path_to_write_images = path_to_write_images / 'train'
            path_to_write_lbls = path_to_write_lbls / 'train'

        img_files = glob(str(Path(sub_folder) / 'images/*.jpg')) + glob(str(Path(sub_folder) / 'images/*.png'))
        #lbl_files = glob(str(Path(sub_folder) / 'lables/*')) # NAOR IS AN IDIOT he wrote lables and not labels.

        for img_path in img_files:
            shutil.copy(img_path, path_to_write_images / (str(Path(img_path).stem) +'.png'))
            lbl_path = Path(img_path).parents[1] / 'lables' / (str(Path(img_path).stem) +'.txt')
            if lbl_path.exists():
                #lbl_path = Path(img_path).parents[2] / 'labels' / str(Path(img_path).stem) +'.txt'
                shutil.copy(lbl_path, path_to_write_lbls / lbl_path.name)
            else:
                print("No label to this image: %s" % img_path)



if __name__ == '__main__':
    from_all_in_one()
