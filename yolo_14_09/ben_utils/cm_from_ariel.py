import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
from pathlib import Path
from tqdm  import tqdm
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from datetime import date


today = date.today()

def plot_confusion_matrix(class_num,cm, title='Confusion matrix', cmap=plt.cm.Oranges):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    names = [ 'Person', 'Person on board a vessel', 'Swimmer', 'Sail boat', 'Floating object',  'Dvora', 'Zeara', 'PWC', 'Merchant Ship', 'Inflatable Boat', 'Vessel']
    plt.title(title)
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    ax = plt.gca()
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[1]), range(cm.shape[0])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('True label ')
    plt.ylabel('Predicted label')
    plt.savefig(out_dir+'CM_area_'+str(date.today()))
    plt.show(block=True)


def bb_intersection_over_union(boxAa,boxBb):

    # determine the (x, y)-coordinates of the intersection rectangle
    # Carmi recomends on using pixel coordinates
    boxA, boxB = [0,0,0,0],[0,0,0,0]
    epsilon = 0.001
    boxAw, boxAh, boxBw, boxBh = boxAa[2], boxAa[3], boxBb[2], boxBb[3]
    boxAw, boxAh, boxBw, boxBh = boxAa[2], boxAa[3], boxBb[2], boxBb[3]
    boxA[0], boxA[1] = boxAa[0]-0.5*boxAw+epsilon, boxAa[1]-0.5*boxAh
    boxA[2], boxA[3] = boxAa[0]+0.5*boxAw-epsilon, boxAa[1]+0.5*boxAh
    boxB[0], boxB[1] = boxBb[0]-0.5*boxBw+epsilon, boxBb[1]-0.5*boxBh
    boxB[2], boxB[3] = boxBb[0]+0.5*boxBw-epsilon, boxBb[1]+0.5*boxBh
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou




def score_calc(labels, preds, neg_iou_thd): #  several obj in frame


    df = pd.DataFrame(columns=['img_file', 'file_idx', 'bbox', 'area', 'iou', 'score', 'class_gt', 'class_pred', 'bbox_sum'])

    for i, pred_row in tqdm(preds.iterrows()):

        max_iou = 0
        max_iou_idx = 0
        #select pred files according to gt files
        image_labels_df = labels.loc[labels['file_idx'] == pred_row.file_idx]

        for idx, label_row in image_labels_df.iterrows():
                #calculate iou between gt to pred
            iou = bb_intersection_over_union(label_row.bbox, pred_row.bbox)
            if not label_row.file_idx in list(preds['file_idx']):  # no detections for image
                df = df.append({'img_file': label_row.img_file, 'file_idx': label_row.file_idx,
                                'bbox': label_row.bbox, 'area': label_row.area,
                                'iou': 0, 'score': 0, 'class_gt': label_row.cls_id, 'class_pred' : 6, 'bbox_sum': 0}, ignore_index=True)

            if not label_row.cls_id in list(preds.cls_id):  # no detections for object
                df = df.append({'img_file': label_row.img_file, 'file_idx': label_row.file_idx,
                                'bbox': label_row.bbox, 'area': label_row.area,
                                'iou': 0, 'score': 0, 'class_gt': label_row.cls_id,
                                'class_pred' : 6, 'bbox_sum': 0}, ignore_index=True)

            if iou > max_iou:
                max_iou = iou
                max_iou_idx = idx
                class_pred = pred_row.cls_id
                pred_row_bbox = pred_row.bbox
                pred_row_area = pred_row.area
                score_pred = pred_row.score
                pred_row_bbox_sum = pred_row.bbox.sum()
                class_gt = label_row.cls_id
                # image_preds_df.drop(idx, inplace=True)

        if max_iou >= neg_iou_thd:
            df = df.append({'img_file': label_row.img_file, 'file_idx': label_row.file_idx,
                            'bbox': pred_row_bbox, 'area': pred_row_area,
                            'iou': max_iou, 'score': score_pred,
                            'class_gt': class_gt, 'class_pred': class_pred,
                            'bbox_sum': pred_row_bbox_sum}, ignore_index=True)

    class_num = df.class_pred.nunique()
    print('number of classes: ', class_num)
    y_pred = np.array(df.class_pred)
    y_test = np.array(df.class_gt)
    cm_matrix = confusion_matrix(y_test, y_pred.astype(np.float64), normalize=None)
    np.save('cm_sample', cm_matrix)
    plot_confusion_matrix(class_num,cm_matrix, title='Confusion matrix', cmap=plt.cm.Oranges)
    return df, cm_matrix


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
        # save failed images to directory, good for error analysis


def print_specs(partition, n_ims, f_ims, iou, score):
    print('-'*20)
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
#    if iou = 0:


    return f'{n_ims-f_ims}/{n_ims}',(n_ims-f_ims)/n_ims * 100,iou * 100, score * 100
# release this function since I do not care about partition!

def create_multiple_preds_df(path2labels, idx, df2merge, start_file_num, end_file_num):
    files = glob.glob(f'{path2labels}/*.txt')[start_file_num:end_file_num]

    if not type(df2merge) == bool:
        df = df2merge
    else:
        df = pd.DataFrame(columns=["img_file", "bbox", "area", "cls_id", "score", "file_idx"])
    for label_file in tqdm(files):
        if os.path.getsize(label_file):
            data = pd.read_csv(label_file, header=None, sep=" ").to_numpy()
        else:
            continue
        img_file = Path(label_file).stem #final path compoment with out its suffix

        for line in data:
            bbox = line[1:5]
            area = bbox[2]*bbox[3]# area should be as yoloformat w*h(line[3]-line[1])*(line[4]-line[2])
            cls_id = line[0]
            score = 1#line[5]
            file_idx = f'{img_file}_{idx}'
            df = df.append({"img_file": img_file, "bbox": bbox, "area": area, "cls_id": cls_id, "score": score,
                            "file_idx": file_idx},
                           ignore_index=True)
    return df


def create_multiple_labels_df(path2labels, image_size, idx, df2merge, start_file_num, end_file_num):
    files = glob.glob(fr'{path2labels}/*.txt')[start_file_num:end_file_num]
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
            orig_bbox = line[1:]#yolo_to_orig_coords(line[1:], image_size)
            bbox = orig_bbox
            area = bbox[2]*bbox[3]#(orig_bbox[2]-orig_bbox[0])*(orig_bbox[3]-orig_bbox[1])
            cls_id = line[0]
            file_idx = f'{img_file}_{idx}'

            df = df.append({"img_file": img_file, "bbox": bbox, "area": area, "cls_id": cls_id, "file_idx": file_idx},
                           ignore_index=True)

    return df



def run_calculations(start_file_num, end_file_num, out_dir, imageOutputPath, prediction_labels_folder, true_labels_folder, images_path,
                     size_partitions=False, save_bbox_flag=False, neg_iou_thd=0.2, pos_iou_thd=0.4):
    size_partitions = size_partitions
    start_file_num, end_file_num = start_file_num, end_file_num

    if glob.glob(f'{images_path}/*.png'):
        imgs = glob.glob(f'{images_path}/*.png')  # jpg

        image_size = cv2.imread(imgs[0]).shape[:2]
        predictions_df = create_multiple_preds_df(prediction_labels_folder, 0, False, start_file_num, end_file_num)
        true_labels_df = create_multiple_labels_df(true_labels_folder, image_size, 0, False, start_file_num, end_file_num)

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

    print('\ntrue labels head: ', true_labels_df.head(), '\nprediction labels head: ', predictions_df.head())
    all_scores,_ = score_calc(true_labels_df, predictions_df, neg_iou_thd)
    print('done calculation')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(imageOutputPath, exist_ok=True)
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
            # plt.savefig(f'{out_dir}/failes_{size[0]}.jpg')
#        scores.loc[scores['score'] <= pos_iou_thd].to_csv(f'{out_dir}/no_detections_{size[0]}.csv')
        # Low score filter
        if len(scores[(scores['score'] > neg_iou_thd) & (scores['score'] < pos_iou_thd)]) > 0:
            scores[(scores['score'] > neg_iou_thd) & (scores['score'] < pos_iou_thd)].hist(bins=10)
            plt.suptitle(f'Low score details [0.5, 0.6]', fontsize=20)
            # plt.savefig(f'{out_dir}/low_score_info_{size[0]}.jpg')
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
    out_dir = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\check_res\ariel'
    imageOutputPath = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\check_res\ariel\image_outpath'
    if not ((os.path.exists(out_dir)) or (os.path.exists(imageOutputPath))):
        os.mkdir(out_dir),os.mkdir(imageOutputPath)
    neg_iou_thd = 0.001
    prediction_labels_folder = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\runs\val\val_on_sliced_23_12\labels'
    #prediction_labels_folder = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\runs\val\val_on_full_images_23_12\labels'
    true_labels_folder = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_23_12_2021\sliced\labels\val'
    start_file_num = 0
    end_file_num = len(glob.glob(os.path.join(true_labels_folder, '*.txt')))
    today = date.today()
    today = '_'.join((str(today).split('-')[-1],str(today).split('-')[-2],str(today).split('-')[0]))
    up_area_thr, down_area_thr = 1.1, 0.9

    images_path = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_23_12_2021\sliced\images\val'
    #images_path = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_23_12_2021\images\val'
    size_partitions = [(0,1)]
    res = run_calculations(start_file_num, end_file_num, out_dir, imageOutputPath, prediction_labels_folder, true_labels_folder, images_path,
                     size_partitions, save_bbox_flag=False, neg_iou_thd=neg_iou_thd)
    indices = [f'from: {i[0]}' if len(i) == 1 else f'{i[0]} to {i[1]}' for i in size_partitions]
    res = np.array(res)
    samples = res[:, 0]
    tpr = res[:, 1]
    mean_iou = res[:, 2]
    mean_score = res[:, 3]
    dict_to_frame = {'Range': indices, 'Sampels': samples, 'tpr': tpr, 'mean iou': mean_iou, 'mean_score': mean_score}
    df = pd.DataFrame.from_dict(dict_to_frame)
    # df.to_csv('U:\\yolov5\\runs\\detect\\5mix_single_chirp_updown_1.5Kmodel_0.05depth\\out_dir\\'+str(today)+'_.csv')


