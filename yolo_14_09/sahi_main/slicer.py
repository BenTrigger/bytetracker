import os
import sys
import time
import numpy as np
import pandas as pd
from sahi_main.sahi.slicing_offline import slice_image_mulriproc
from sahi_main.sahi.prediction import ObjectPrediction
from typing import List, Union
import cv2
from sahi.postprocess.combine import UnionMergePostprocess, PostprocessPredictions, NMSPostprocess
import copy
from pathlib import Path
# import itertools
import glob
from detection_infer_v1_multi_offline import YoloDetection


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def slice_img_nd_label(
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
                mystring = str(
                    "8 " + str(truncate(float(p[0]), 7)) + " " + str(truncate(float(p[1]), 7)) + " "
                    + str(truncate(float(p[2]), 7)) + " " + str(truncate(float(p[3]), 7)))
                myfile.write(mystring)
                myfile.write("\n")
        myfile.close()


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
        slice_img_nd_label(path_to_images, weights_path=weights_path, output_path=outputs_path,
                   save_images_results=save_images_results, save_slices=save_slices)

    else:
        for image in glob.glob(f'{path_to_images}/*.jpg'):
            slice_img_nd_label(image, weights_path=weights_path, output_path=outputs_path, yv5=yv5,
                       save_images_results=save_images_results, save_slices=save_slices)


if __name__ == '__main__':
    run(path_to_images='/MyHomeDir/datasets/test/offline_test/', weights_path='/MyHomeDir/yolov5_master_camri_slices/weights/best_6epochs_1024_smd_hous1_charls0.pt',
        outputs_path='/MyHomeDir/datasets/test/offline_test/', save_images_results=True, save_slices=True)

