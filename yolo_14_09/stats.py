import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
def bb_intersection_over_union(boxA, boxB):
    # boxA = [boxA[0], boxA[1], boxA[0]+boxA[2], boxA[1]+boxA[3]]
    # boxB = [boxB[0], boxB[1], boxB[0]+boxB[2], boxB[1]+boxB[3]]
    boxA = list(map(lambda x:float(x),boxA))
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(float(boxA[0]), boxB[0])
    yA = max(float(boxA[1]), boxB[1])
    xB = min(float(boxA[2]), boxB[2])
    yB = min(float(boxA[3]), boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def score_calc(labels, preds, neg_iou_thd):
    # d = {'img_file': [], 'bbox': [], 'area': [], 'iou': [], 'score': [], 'success': []}
    # df = pd.DataFrame(data=d)
    df = pd.DataFrame(columns=['img_file', 'file_idx', 'bbox', 'area', 'iou', 'score', 'success'])
    for i, label_row in tqdm(labels.iterrows()):
        if not label_row.file_idx in list(preds['file_idx']):  # no detections for image
            df = df.append({'img_file': label_row.img_file, 'file_idx': label_row.file_idx,
                            'bbox': label_row.bbox, 'area': label_row.area,
                            'iou': 0, 'score': 0, 'success': 0}, ignore_index=True)

        else:
            max_iou = 0
            # indices = [k for k, x in enumerate(preds['img_file']) if x == gt_img]
            image_preds_df = preds.loc[preds['file_idx'] == label_row.file_idx]
            max_iou_idx = 0
            for idx, pred_row in image_preds_df.iterrows():
                dt_bbx = list(pred_row.bbox)
                iou = bb_intersection_over_union(label_row.bbox, pred_row.bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = idx
            if max_iou >= neg_iou_thd:
                df = df.append({'img_file': label_row.img_file, 'file_idx': label_row.file_idx, 'bbox': label_row.bbox,
                                'area': label_row.area, 'iou': max_iou, 'score': preds['score'][max_iou_idx],
                                'success': 1}, ignore_index=True)

            else: #max_iou < neg_iou_thd:
                df = df.append({'img_file': label_row.img_file, 'file_idx': label_row.file_idx, 'bbox': label_row.bbox,
                                'area': label_row.area, 'iou': max_iou, 'score': preds['score'][max_iou_idx],
                                'success': 0}, ignore_index=True)

    return df



# def score_calc(labels, preds, neg_iou_thd):
#     # d = {'img_file': [], 'bbox': [], 'area': [], 'iou': [], 'score': [], 'success': []}
#     # df = pd.DataFrame(data=d)
#     df = pd.DataFrame(columns=['img_file', 'bbox', 'area', 'iou', 'score', 'success'])
#     for i, label_row in labels.iterrows():
#         if not label_row.img_file in list(preds['img_file']):  # no detections for image
#             df = df.append({'img_file': label_row.img_file, 'bbox': label_row.bbox, 'area': label_row.area,
#                             'iou': 0, 'score': 0, 'success': 0}, ignore_index=True)
#
#         else:
#             max_iou = 0
#             # indices = [k for k, x in enumerate(preds['img_file']) if x == gt_img]
#             image_preds_df = preds.loc[preds['img_file'] == label_row.img_file]
#             max_iou_idx = 0
#             for idx, pred_row in image_preds_df.iterrows():
#                 dt_bbx = list(pred_row.bbox)
#                 iou = bb_intersection_over_union(label_row.bbox, pred_row.bbox)
#                 if iou > max_iou:
#                     max_iou = iou
#                     max_iou_idx = idx
#             if max_iou >= neg_iou_thd:
#                 df = df.append({'img_file': label_row.img_file, 'bbox': label_row.bbox, 'area': label_row.area,
#                                 'iou': max_iou, 'score': preds['score'][max_iou_idx], 'success': 1}, ignore_index=True)
#
#             else: #max_iou < neg_iou_thd:
#                 df = df.append({'img_file': label_row.img_file, 'bbox': label_row.bbox, 'area': label_row.area,
#                                 'iou': max_iou, 'score': preds['score'][max_iou_idx], 'success': 0}, ignore_index=True)
#
#     return df


def save_bbox(failed_imgs, failed_bbxs, imgs_dir, imageOutputPath):
    color = (0, 255, 0)
    for i, failed_img in enumerate(failed_imgs):
        imageData = cv2.imread(f'{imgs_dir}/{failed_img}')
        x0, y0, x1, y1 = failed_bbxs[i][0], failed_bbxs[i][1], failed_bbxs[i][2], failed_bbxs[i][3]
        if x0 < 0: 
            x0 = 0
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 900)
        cv2.rectangle(imageData, (int(x0), int(y0), int(x1), int(y1)), color, thick)
        cv2.putText(imageData, 'boat', (int(x0), int(y0) - 12), 0, 1e-3 * imgHeight, color, thick//3)
        cv2.imwrite(f'{imageOutputPath}/{i}.jpg', imageData)


def print_specs(partition, n_ims, f_ims, iou, score):
    print('-'*20)
    print(f'partition: {partition}')
    print(f'n_ims: {n_ims}')
    print(f'f_ims: {f_ims}')
    print(f'iou: {iou}')
    print(f'score: {score}')

    if len(partition) == 1:
        print('pixel object area from:', partition[0])
    else:
        print('pixel object area from:', partition[0], 'to:', partition[1])
    if n_ims:
        print('TPR:', f'{n_ims-f_ims}/{n_ims}', (n_ims-f_ims)/n_ims * 100, '%')

        print('mean iou:', iou * 100)
        print('mean score:', score * 100)
    else:
        print('no images in this category')

    return f'{n_ims-f_ims}/{n_ims}',(n_ims-f_ims)/n_ims * 100,iou * 100, score * 100


def create_multiple_preds_df(path2labels, idx, df2merge):
    files = glob.glob(f'{path2labels}/*.txt')
    if not type(df2merge) == bool:
        df = df2merge
    else:
        df = pd.DataFrame(columns=["img_file", "bbox", "area", "cls_id", "score", "file_idx"])
    for label_file in tqdm(files):
        if os.path.getsize(label_file):
            data = pd.read_csv(label_file, header=None, sep=" ").to_numpy()
        else:
            continue
        img_file = Path(label_file).stem
        # if not type(data[0]) == list:
        #     data = [data]
        for line in data:
            bbox = line[1:5]
            area = (line[3]-line[1])*(line[4]-line[2])
            cls_id = line[0]
            score = 0.5# line[5]
            file_idx = f'{img_file}_{idx}'
            df = df.append({"img_file": img_file, "bbox": bbox, "area": area, "cls_id": cls_id, "score": score,
                            "file_idx": file_idx},
                           ignore_index=True)
    return df


def create_multiple_labels_df(path2labels, image_size, idx, df2merge):
    files = glob.glob(fr'{path2labels}/*.txt')
    if not type(df2merge) == bool:
        df = df2merge
    else:
        df = pd.DataFrame(columns=["img_file", "bbox", "area", "cls_id"])
    for label_file in tqdm(files):
        if os.path.getsize(label_file):
            data = pd.read_csv(label_file, header=None, sep=" ").to_numpy()
        else:
            continue
        img_file = Path(label_file).stem
        for line in data:
            orig_bbox = yolo_to_orig_coords(line[1:], image_size)
            bbox = orig_bbox
            area = (orig_bbox[2]-orig_bbox[0])*(orig_bbox[3]-orig_bbox[1])
            cls_id = line[0]
            file_idx = f'{img_file}_{idx}'

            df = df.append({"img_file": img_file, "bbox": bbox, "area": area, "cls_id": cls_id, "file_idx": file_idx},
                           ignore_index=True)

    return df


def create_preds_df(path2labels):
    files = glob.glob(fr'{path2labels}/*.txt')
    df = pd.DataFrame(columns=["img_file", "bbox", "area", "cls_id", "score"])
    for label_file in files:
        if os.path.getsize(label_file):
            data = pd.read_csv(label_file, header=None, sep=" ").to_numpy()
        else:
            continue
        img_file = Path(label_file).stem
        # if not type(data[0]) == list:
        #     data = [data]
        for line in data:
            bbox = line[1:5]
            area = (line[3]-line[1])*(line[4]-line[2])
            cls_id = line[0]
            score = line[5]
            df = df.append({"img_file": img_file, "bbox": bbox, "area": area, "cls_id": cls_id, "score": score},
                           ignore_index=True)
    return df


def create_labels_df(path2labels, image_size):
    files = glob.glob(fr'{path2labels}/*.txt')
    df = pd.DataFrame(columns=["img_file", "bbox", "area", "cls_id"])
    for label_file in files:
        # data = np.loadtxt(label_file)
        # data = pd.read_csv(label_file, header=None, sep=" ").to_numpy()
        if os.path.getsize(label_file):
            data = pd.read_csv(label_file, header=None, sep=" ").to_numpy()
        else:
            continue
        img_file = Path(label_file).stem
        # if not type(data[0]) == list:
        #     data = [data]
        for line in data:
            orig_bbox = yolo_to_orig_coords(line[1:], image_size)
            bbox = orig_bbox
            area = (orig_bbox[2]-orig_bbox[0])*(orig_bbox[3]-orig_bbox[1])
            cls_id = line[0]
            df = df.append({"img_file": img_file, "bbox": bbox, "area": area, "cls_id": cls_id},
                           ignore_index=True)

    return df


def yolo_to_orig_coords(box, image_size):
    dh, dw = image_size
    x, y, w, h = box
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1
    x_min, y_min = l, t
    x_max, y_max = r, b
    return [x_min, y_min, x_max, y_max]


def run_calculations(out_dir, imageOutputPath, prediction_labels_folder, true_labels_folder, images_path,
                     size_partitions=False, save_bbox_flag=False, neg_iou_thd=0.2, pos_iou_thd=0.4):

    size_partitions = size_partitions if size_partitions else [(0, 1500), (1500, 2000), (2000, 2500), (2500, 3000),
                                                               (3000, 4000), (10000, 20000), (20000,)]
    if glob.glob(f'{images_path}/*.png'):
        imgs = glob.glob(f'{images_path}/*.png')  # jpg

        image_size = cv2.imread(imgs[0]).shape[:2]
        predictions_df = create_multiple_preds_df(prediction_labels_folder, 0, False)
        true_labels_df = create_multiple_labels_df(true_labels_folder, image_size, 0, False)

    else:
        imgs = []
        for folder in os.listdir(images_path):
            imgs.extend(glob.glob(f'{images_path}/{folder}/*.png'))  # jpg
            image_size = cv2.imread(imgs[0]).shape[:2]

        for j, folder in enumerate(os.listdir(prediction_labels_folder)):
            if j == 0:
                predictions_df = False
            predictions_df = create_multiple_preds_df(f'{prediction_labels_folder}/{folder}', j, predictions_df)

        del folder

        for k, folder in enumerate(os.listdir(images_path)):
            if k == 0:
                true_labels_df = False
            true_labels_df = create_multiple_labels_df(f'{true_labels_folder}/{folder}', image_size,  k, true_labels_df)

    print(true_labels_df.head(), predictions_df.head())
    all_scores = score_calc(true_labels_df, predictions_df, neg_iou_thd)
    print('done calculation')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(imageOutputPath, exist_ok=True)
    miss_pct = []
    stats = []
    for i, size in enumerate(size_partitions):
        if len(size) == 1:
            scores = all_scores.loc[(all_scores['area'] >= size[0])]
        else:
            scores = all_scores.loc[(all_scores['area'] >= size[0]) & (all_scores['area'] <= size[1])]

        fig = plt.figure(i)
        if len(scores.loc[scores['iou'] <= neg_iou_thd]['area']):
            scores.loc[scores['iou'] <= neg_iou_thd]['area'].hist(bins=5)
            plt.suptitle(f'Failes at IOU <= {neg_iou_thd} ', fontsize=20)
            plt.xlabel('area = WxH (#pixels)', fontsize=18)
            plt.ylabel('# Objects', fontsize=16)
            fig.savefig(f'{out_dir}/failes_{size[0]}.jpg')
        # plt.close('all')
        # remember that in SMD train and test there are NIR images ***
        # save csv of images that were not detected at all
        scores.loc[scores['score'] <= pos_iou_thd].to_csv(f'{out_dir}/no_detections_{size[0]}.csv')
        # Low score filter
        if len(scores[(scores['score'] > neg_iou_thd) & (scores['score'] < pos_iou_thd)]) > 0:
            scores[(scores['score'] > neg_iou_thd) & (scores['score'] < pos_iou_thd)].hist(bins=10)
            plt.suptitle(f'Low score details [0.5, 0.6]', fontsize=20)
            plt.savefig(f'{out_dir}/low_score_info_{size[0]}.jpg')
            plt.close('all')

        failed_imgs = scores.loc[scores['iou'] <= pos_iou_thd]['img_file'].to_list()
        failed_bbxs = scores.loc[scores['iou'] <= pos_iou_thd]['bbox'].to_list()

        if save_bbox_flag:
            save_bbox(failed_imgs, failed_bbxs, images_path, imageOutputPath)

        total_images = scores['img_file']

        res = print_specs(size, n_ims=len(total_images), f_ims=len(failed_imgs), iou=scores['iou'].mean(),
                    score=scores['score'].mean())
        stats.append(res)
    return stats


if __name__ == '__main__':
    save_bbox_flag = False
    out_dir = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_13_10_21\output'
    imageOutputPath = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_13_10_21\output\img'
    neg_iou_thd = 0.2
    prediction_labels_folder = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\runs\detect\detect_val_10_pic\labels'
    true_labels_folder = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_13_10_21\labels\val'
    images_path = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_13_10_21\images\val'
    size_partitions = [(0,)]#[(0, 15*15), (15*15, 25*25), (25*25, 50*50), (50*50, 70*70), (70*70, 100*100), (10000,)]
    res = run_calculations(out_dir, imageOutputPath, prediction_labels_folder, true_labels_folder, images_path,
                     size_partitions, save_bbox_flag=False, neg_iou_thd=0.2)
    indices = [f'from: {i[0]}' if len(i) == 1 else f'{i[0]} to {i[1]}' for i in size_partitions]
    res = np.array(res)
    samples = res[:, 0]
    tpr = res[:, 1]
    mean_iou = res[:, 2]
    mean_score = res[:, 3]
    dict_to_frame = {'Range': indices, 'Sampels': samples, 'tpr': tpr, 'mean iou': mean_iou, 'mean_score': mean_score}
    df = pd.DataFrame.from_dict(dict_to_frame)
    df.to_csv('./stats.csv')

    # need to get from real and predict normal vector/df
    # print("Classification Report")
    # print(classification_report(y_t, predicted_model))
    # print("Confusion Report")
    # cm = confusion_matrix(y_t, predicted_model)
    # print(cm)



