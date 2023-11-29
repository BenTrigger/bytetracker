from pathlib import Path
from glob import glob
from tqdm import tqdm
import shutil
from random import seed
from random import random
seed(1)


def main(trgt_imgs, dest_folder, split_to_train_val=False, skip_frames_counter=10):
    counter = 1
    for img_path in tqdm(trgt_imgs):
        if counter % skip_frames_counter != 0:
            counter += 1
            continue
        counter = 1

        # COPY img and val from one source

        folder_dest = 'train' # train or test
        if split_to_train_val and random() < 0.10:  # 10% val set
            folder_dest = 'val'

        movie_name_folder = Path(img_path).parent.parent.name
        tmp_img_dest = Path(dest_folder) / 'images' / folder_dest / Path(movie_name_folder + '_' + (str(Path(img_path).stem) +'.png'))
        shutil.copy(img_path, tmp_img_dest)
        lbl_path = Path(img_path).parents[1] / 'labels' / (str(Path(img_path).stem) +'.txt')
        if lbl_path.exists():
            tmp_lbl_dest = Path(dest_folder) / 'labels' / folder_dest / Path(movie_name_folder + '_' + lbl_path.name)
            shutil.copy(lbl_path, tmp_lbl_dest)
        else:
            print("No label to this image: %s" % lbl_path)

        # move files to val
        # if split_to_train_val and random() < 0.10: # 10% val set
        #     lbl_source_folder = Path(dest_folder).parents[1] / 'labels' / 'train'
        #     lbl_file_source = Path(lbl_source_folder) / str(Path(img_path).stem + '.txt')
        #     lbl_file_dest = Path(lbl_source_folder).parents[0] / 'val' / Path(lbl_file_source).name
        #     img_file_dest = Path(dest_folder) / str(Path(img_path).stem + '.jpg')
        #
        #     shutil.move(img_path, img_file_dest)
        #     shutil.move(lbl_file_source, lbl_file_dest)


if __name__ == '__main__':
    trgt_imgs = glob(r'Z:\yolo_14_09\data\Baloons_DATA\retagged_7_8_22\Detection_Training_Datasets\Ibdis*\images\*')
    #trgt_imgs = glob(r'/MyHomeDir/yolo_14_09/data/Baloons_DATA/retagged_7_8_22/SECOND_PART_TRAIN_DataSet/*/images/*')
    dest = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\extra_images_10_7_23'
    #Path(dest).mkdir(parents=True, exist_ok=True)
    #dest = r'/MyHomeDir/yolo_14_09/data/Baloons_DATA/tagged_18_10_22_one_to_ten'
    main(trgt_imgs, dest, split_to_train_val=True, skip_frames_counter=1) # KEEP ALL FRAMES = 1

