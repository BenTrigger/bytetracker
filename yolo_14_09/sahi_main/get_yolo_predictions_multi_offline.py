import os
import sys
import time
import numpy as np
import pandas as pd

from sahi_main.sahi.slicing_offline import slice_image_mulriproc
from sahi_main.sahi.prediction import ObjectPrediction
from typing import List, Union
import cv2
from sahi_main.sahi.postprocess.combine import UnionMergePostprocess, PostprocessPredictions, NMSPostprocess
import copy
from pathlib import Path
# import itertools
import glob
from detection_infer_v1_multi_offline import YoloDetection

# detection_model = Yolov5DetectionModel(
#     model_path='',
#     config_path='',
#     prediction_score_threshold=0.4,
#     device="cuda"
# )
#
# image = read_image()
# # b = get_sliced_prediction():
# """"
#     Returns:
#         sliced_image_result: SliceImageResult:
#                                 sliced_image_list: list of SlicedImage
#                                 image_dir: str
#                                     Directory of the sliced image exports.
#                                 original_image_size: list of int
#                                     Size of the unsliced original image in [height, width]
#         num_total_invalid_segmentation: int
#             Number of invalid segmentation annotations.
# """
# sliced_ims = slice_image(
#         image=image,
#         slice_height=1024,
#         slice_width=1024,
#         overlap_height_ratio=0.2,
#         overlap_width_ratio=0.2,
#     )
#
# # elif postprocess_type == "NMS":
# postprocess = NMSPostprocess(
#     match_threshold=postprocess_match_threshold,
#     match_metric=postprocess_match_metric,
#     class_agnostic=postprocess_class_agnostic,
# )


class NMSPostprocess:
    def __init__(self, match_threshold, match_metric, class_agnostic):
        self.match_threshold = match_threshold
        self.match_metric = match_metric
        self.class_agnostic = class_agnostic

    def __call__(self, object_predictions):
        source_object_predictions: List[ObjectPrediction] = copy.deepcopy(object_predictions)
        selected_object_predictions: List[ObjectPrediction] = []
        while len(source_object_predictions) > 0:
            # select object prediction with highest score
            df = pd.DataFrame(source_object_predictions)
            df.sort_values('score', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            selected_object_prediction = df.loc[0]
            df.drop(0, inplace=True)
            # remove selected prediction from source list
            # del source_object_predictions[0]
            # if any element from remaining source prediction list matches, remove it
            new_source_object_predictions = []
            for candidate_object_prediction in df.iterrows():
                if not self.has_match(selected_object_prediction, candidate_object_prediction[1]):
                    new_source_object_predictions.append(candidate_object_prediction[1])
            source_object_predictions = new_source_object_predictions
            # append selected prediction to selected list
            selected_object_predictions.append(selected_object_prediction)
        return selected_object_predictions

    def has_match(self, pred1, pred2):
        threshold_condition = self.calculate_bbox_iou(pred1, pred2) > self.match_threshold
        category_condition = pred1.category == pred2.category or self.class_agnostic
        return threshold_condition and category_condition

    @staticmethod
    def calculate_bbox_ios(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
        """Returns the ratio of intersection area to the smaller box's area"""
        box1 = np.array(pred1.bbox[0])
        box2 = np.array(pred2.bbox[0])
        area1 = calculate_area(box1)
        area2 = calculate_area(box2)
        intersect = calculate_intersection_area(box1, box2)
        smaller_area = np.minimum(area1, area2)
        if smaller_area:
            return intersect / smaller_area
        else:
            return 0

    @staticmethod
    def calculate_bbox_iou(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
        """Returns the ratio of intersection area to the union"""
        box1 = np.array(pred1.bbox[0])
        box2 = np.array(pred2.bbox[0])
        area1 = calculate_area(box1)
        area2 = calculate_area(box2)
        intersect = calculate_intersection_area(box1, box2)
        if (area1 + area2 - intersect):
            return intersect / (area1 + area2 - intersect)
        else:
            return 0


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def bbox_to_yolo(bbox, size):
    width, height = size
    dw = 1. / width
    dh = 1. / height

    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]

    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2

    w = xmax - xmin
    h = ymax - ymin

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return [x, y, w, h]


def calculate_intersection_area(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Args:
        box1 (np.ndarray): np.array([x1, y1, x2, y2])
        box2 (np.ndarray): np.array([x1, y1, x2, y2])
    """
    left_top = np.maximum(box1[:2], box2[:2])
    right_bottom = np.minimum(box1[2:], box2[2:])
    width_height = (right_bottom - left_top).clip(min=0)
    return width_height[0] * width_height[1]


def calculate_area(box: Union[List[int], np.ndarray]) -> float:
    """
    Args:
        box (List[int]): [x1, y1, x2, y2]
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def predict_4k(
    image,
    detection_model=None,
    slice_height: int = 256,
    slice_width: int = 256,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "UNIONMERGE",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,
    weights_path='',
    output_path='',
    yv5=False,
    save_images_results=False,
    save_slices=False
):
    if not yv5:
        yv5 = YoloDetection(weights=weights_path, cuda='')
    t0 = time.time()
    filename = Path(image).stem
    # create slices from full image
    t_slicing_sahi = time.time()
    if not save_slices:
        slices_output_path = False

    slices, starting_pixels = slice_image_mulriproc(
        image=image,
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        output_dir=save_slices if not save_slices else output_path,
        output_file_name=filename
    )

    # # check for double files:
    # tmp = [x for x in range(len(batch))]
    # all_combos = itertools.permutations(tmp, 2)
    # for combo in all_combos:
    #     if np.array_equal(batch[combo[0]], batch[combo[1]]):
    #         print('two pics identical!!!!')

    print(f'time for sahi slicing = {time.time() - t_slicing_sahi} for 4k image')

    num_slices = slices.shape[0]

    #print("Number of slices:", num_slices)
    preds, sliced_preds = yv5.run_inference(starting_pixels, slices)

    os.makedirs(f'{output_path}labels/', exist_ok=True)
    if os.path.isfile(f'{output_path}labels/{filename}.txt'):
        os.remove(f'{output_path}labels/{filename}.txt')

    for file_idx, slice_preds in enumerate(sliced_preds):
        with open(f'{output_path}labels/{filename}_{file_idx}.txt', 'a') as myfile:
            for p in slice_preds:
                box = bbox_to_yolo(p, (1024,1024))
                mystring = str(
                    "8 " + str(truncate(float(box[0]), 7)) + " " + str(truncate(float(box[1]), 7)) + " "
                    + str(truncate(float(box[2]), 7)) + " " + str(truncate(float(box[3]), 7)))
                myfile.write(mystring)
                myfile.write("\n")
        myfile.close()

    nmspps = NMSPostprocess(match_threshold=postprocess_match_threshold, match_metric=postprocess_match_metric,
                            class_agnostic=postprocess_class_agnostic,)
    full_object_prediction_list = nmspps(preds)

    # for pred

    print(f'Done. ({time.time() - t0:.3f}s)')


    if save_images_results:
        visualize_object_predictions(image, full_object_prediction_list, output_dir=f'{output_path}/preds/',
                                     file_name=filename)


def visualize_object_predictions(
        image: np.array,
        object_prediction_list,
        rect_th: float = 1,
        text_size: float = 0.3,
        text_th: float = 1,
        color: tuple = (0, 255, 0),
        output_dir: [str] = None,
        file_name: str = "prediction_visual",
        export_format: str = "jpg",
):
    """
    Visualizes prediction category names, bounding boxes over the source image
    and exports it to output folder.
    Arguments:
        object_prediction_list: a list of prediction.ObjectPrediction
        rect_th: rectangle thickness
        text_size: size of the category name over box
        text_th: text thickness
        color: annotation color in the form: (0, 255, 0)
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # select random color if not specified
    # add bbox and mask to image if present
    for object_prediction in object_prediction_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.copy()

        bbox = np.array(object_prediction.bbox[0], dtype=int)
        category_name = 'boat' if object_prediction.category == 8 else print('problem')
        score = object_prediction.score

        # visualize masks if present
        cv2.rectangle(
            image,
            tuple(bbox[0:2]),
            tuple(bbox[2:4]),
            color=color,
            thickness=rect_th,
        )
        # arange bounding box text location
        if bbox[1] - 5 > 5:
            bbox[1] -= 5
        else:
            bbox[1] += 5
        # add bounding box text
        label = "%s %.2f" % (category_name, score)
        cv2.putText(
            image,
            label,
            tuple(bbox[0:2]),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            color,
            thickness=text_th,
        )
    if output_dir:
        # create output folder if not present
        os.makedirs(output_dir, exist_ok=True)
        # save inference result
        save_path = os.path.join(output_dir, file_name + "." + export_format)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}


def run(path_to_images, weights_path, outputs_path, save_images_results, save_slices):
    if not os.path.isdir(path_to_images):
        predict_4k(path_to_images, weights_path=weights_path, output_path=outputs_path,
                   save_images_results=save_images_results, save_slices=save_slices)

    else:
        yv5 = YoloDetection(weights=weights_path, cuda='')
        # for image in glob.glob(f'{path_to_images}/*.JPG'):
        #     predict_4k(image, weights_path=weights_path, output_path=outputs_path, yv5=yv5,
        #                save_images_results=save_images_results, save_slices=save_slices)
        #
        imgs = glob.glob(f'{path_to_images}/*.JPG')
        for i in range(0, len(imgs), 2):
            image = imgs[i]
            predict_4k(image, weights_path=weights_path, output_path=outputs_path, yv5=yv5,
                       save_images_results=save_images_results, save_slices=save_slices)


if __name__ == '__main__':
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-05-02/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-05-02/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-05-03/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-05-03/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-05-06/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-05-06/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-05-10/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-05-10/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-05-11/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-05-11/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-05-18/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-05-18/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-05-22/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-05-22/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-05-23/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-05-23/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-05-24/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-05-24/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-05-30/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-05-30/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-06-03/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-06-03/', save_images_results=False, save_slices=True)
    run(path_to_images='/MyHomeDir/datasets/from_iard/2021-06-17/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/from_iard/sliced/2021-06-17/', save_images_results=False, save_slices=True)
