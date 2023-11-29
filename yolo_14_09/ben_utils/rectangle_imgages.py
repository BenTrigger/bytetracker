from builtins import enumerate

import cv2
import pandas as pd

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

def rectangle_image(prediction_labels_file=None, true_labels_file=None, image_path=None, output_path=None, im=None, preds_as_bbox=None):
    if Path(output_path).exists():
        print("file: %s exists already" % output_path)
        return
    if im is None:
        im = cv2.imread(image_path)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    img_h, img_w, channels = im.shape

    line_thickness = 3
    tl = 1  # line/font thickness
    if preds_as_bbox:
        for val in preds_as_bbox:
            ture_lbl = val['category']
            line = val['bbox'][0]
            cls_id = ture_lbl
            color = colors(cls_id)
            if cls_id :#== 1: if want to rectangle specific labels
                #print(line)
                x1 = int(line[0] * img_w)
                y1 = int(line[1] * img_h)
                x2 = int(line[2] * img_w)
                y2 = int(line[3] * img_h)
                cv2.rectangle(img=im, pt1=(x1, y1), pt2=(x2, y2), color=color)
    elif true_labels_file:
        try:
            data = pd.read_csv(true_labels_file, header=None, sep=" ").to_numpy()
        except Exception as e:
            print(e)
            return
        x1,y1,w1,h1 = 0,0,0,0
        for line in data:
            cls_id = line[0]
            color = colors(cls_id)
            if cls_id is not None :#== 1: if want to rectangle specific labels
                x1 = int(line[1] * img_w)
                y1 = int(line[2] * img_h)
                w1 = int(line[3] * img_w)
                h1 = int(line[4] * img_h)
                c1, c2 = (int(x1-w1/2), int(y1-h1/2)), (int(x1+w1/2), int(y1+h1/2))
                cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if prediction_labels_file:
        try:
            data = pd.read_csv(prediction_labels_file, header=None, sep=" ").to_numpy()
        except Exception as e:
            print(e)
            return
        x2,y2,w2,h2 = 0,0,0,0
        for line in data:
            cls_id = line[0]
            color = colors(cls_id)
            if cls_id :# == 1:  if want to rectangle specific labels
                x2 = int(line[1] * img_w)
                y2 = int(line[2] * img_h)
                w2 = int(line[3] * img_w)
                h2 = int(line[4] * img_h)
                c1, c2 = (int(x2-w2/2), int(y2-h2/2)), (int(x2+w2/2), int(y2+h2/2))
                cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.imwrite(f'{output_path}', im)


if __name__ == "__main__":
    ###option one - ONE FILE TO RECTANGLE

    #prediction_labels_folder = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\images\test\black_panther_27-04-2022_16_45-16_46_cam_2_Frame000173_16_46_11_31_2.txt'
    # prediction_labels_folder = None
    # true_labels_folder = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\labels\test\black_panther_27-04-2022_16_45-16_46_cam_2_Frame000173_16_46_11_31_2.txt'
    # #r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\runs\val\detect_val_20_10_one_pic17\labels\black_panther_04-06-2021_10_10-10_14_cam_1_frame_968.txt'
    # images_path = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\images\test\black_panther_27-04-2022_16_45-16_46_cam_2_Frame000173_16_46_11_31_2.png'
    # output_path = r'Z:\yolo_14_09\data\Baloons_DATA\tagged_7_8_baloons_all\rectangled_black_panther_27-04-2022_16_45-16_46_cam_2_Frame000173_16_46_11_31_2.jpg'
    # rectangle_image(prediction_labels_folder, true_labels_folder, images_path, output_path)
    # exit(1)

    ### option two - ALL FILES IN FOLDER TO RECTANGLE
    # from pathlib import Path
    # from glob import glob
    # from tqdm import tqdm
    # #path_imgs = Path(r'Z:\yolo_14_09\data\Baloons_DATA\tagged_28_07_22_baloons\images\val')  # change it to your path
    # path_imgs = Path(r'Z:\yolo_14_09\data\atr_zafrir\tagged_22_02_22_EREZ_ashdod_youtube\images\val')  # change it to your path
    # # creating output folder
    # #output_path = Path(r'Z:\yolo_14_09\data\atr_zafrir\tagged_28_07_22_Rosh\rectangle_bbox_gt') # swimmers
    # output_path = Path(r'Z:\yolo_14_09\data\atr_zafrir\tagged_22_02_22_EREZ_ashdod_youtube\rectangle_bbox_gt_val_set') # baloons
    # output_path.mkdir(parents=True, exist_ok=True)
    #
    # formats = ['jpg', 'png']
    # img_files = []
    # for ext in formats:
    #     #img_files.extend(glob(f'{path_imgs}\*\images\*.{ext}')) # swimmers
    #     img_files.extend(glob(f'{path_imgs}\*.{ext}')) # swimmers
    #     #img_files.extend(glob(f'{path_imgs}\*\images\*.{ext}')) # baloons
    #
    # set_name = path_imgs.name
    # for img_path in tqdm(img_files):
    #     project_name = Path(img_path).parents[0].name
    #     #lbl_path = Path(img_path).parents[2] / project_name / 'labels' / (str(Path(img_path).stem) +'.txt') # for normal files in project folder
    #     lbl_path = Path(img_path).parents[2] / 'labels' / project_name / (str(Path(img_path).stem) +'.txt') # for swimmers inside train folder format labels\val...
    #     #lbl_path = Path(img_path).parents[2] / 'labels' / str(set_name) / (str(Path(img_path).stem) +'.txt')
    #     # if there is any predictions then get them by enter correct path.
    #
    #
    #     #pred_file = Path(img_path).parents[1] / 'preds' / (str(Path(img_path).stem) +'.txt')
    #     pred_file = None
    #     if lbl_path.exists() and not Path(output_path / (project_name + '_' + Path(img_path).name)).exists():
    #         if pred_file: # with predicted labels
    #             rectangle_image(pred_file, lbl_path, img_path, output_path / (project_name + '_' + Path(img_path).name))
    #         else:          # without predicted labels
    #             rectangle_image(None, lbl_path, img_path, output_path / (project_name + '_' + Path(img_path).name))

    ### option three - ALL FILES IN SUB FOLDERS
    from pathlib import Path
    from glob import glob
    from tqdm import tqdm

    #path_imgs = Path(r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames\extra_images_10_7_23\images\*\*.')  # change it to your path
    path_imgs = Path(r'/MyHomeDir/yolo_14_09/data/Baloons_DATA/tagged_2_2_2023_all_frames/extra_images_10_7_23/images/*/*.')  # change it to your path
    #output_path = Path(r'Z:\yolo_14_09\data\Baloons_DATA\retagged_7_8_22\rectangles_extra_images')  # baloons
    output_path = Path(r'/MyHomeDir/yolo_14_09/data/Baloons_DATA/retagged_7_8_22/rectangles_extra_images')  # baloons

    output_path.mkdir(parents=True, exist_ok=True)

    formats = ['png', 'jpg']
    img_files = []
    for ext in formats:
        #img_files.extend(glob(f'{path_imgs}\*.{ext}')) # swimmers
        img_files.extend(glob(str(path_imgs) + ext))  # swimmers
    set_name = path_imgs.name
    for img_path in tqdm(img_files):
        project_name = Path(img_path).parents[2].name
        lbl_path = Path(img_path).parents[3] / project_name / 'labels' / Path(img_path).parents[0].name / (str(Path(img_path).stem) +'.txt') # for swimmers inside train folder format labels\val...
        #pred_file = Path(img_path).parents[1] / 'preds' / (str(Path(img_path).stem) +'.txt')
        pred_file = None
        folder_out_put = output_path / project_name
        folder_out_put.mkdir(parents=True, exist_ok=True)
        if lbl_path.exists() and not Path(folder_out_put / (project_name + '_' + Path(img_path).name)).exists():
            if pred_file: # with predicted labels
                rectangle_image(pred_file, lbl_path, img_path, folder_out_put / (project_name + '_' + Path(img_path).name))
            else:          # without predicted labels
                rectangle_image(None, lbl_path, img_path, folder_out_put / (project_name + '_' + Path(img_path).name))

