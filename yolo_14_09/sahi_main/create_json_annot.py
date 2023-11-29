import json
# import cv2 as cv
import os
from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import read_image, visualize_object_predictions, ipython_display
from sahi.predict import get_prediction, get_sliced_prediction, predict
import glob

detection_model = Yolov5DetectionModel(
    model_path='../weights/best_6epochs_1024_smd_hous1_charls0.pt',
    prediction_score_threshold=0.4,
    device="cuda",  # or 'cpu'
)



def batch_pred(coco_file,source_image_dir):
    model_name = "Yolov5DetectionModel"
    model_parameters = {
        "model_path": '../weights/best_6epochs_1024_smd_hous1_charls0.pt',
        "device": "cuda",
        "prediction_score_threshold": 0.4,

    }
    apply_sliced_prediction = True
    slice_height = 1024
    slice_width = 1024
    overlap_height_ratio = 0.2
    overlap_width_ratio = 0.2
    export_crop = True

    predict(
        model_name=model_name,
        model_parameters=model_parameters,
        source=source_image_dir,
        coco_file_path=coco_file,
        export_crop=export_crop,
        apply_sliced_prediction=apply_sliced_prediction,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )


path = '/MyHomeDir/datasets/video_frames_merchant'
out_path = '/MyHomeDir/video_frames_merchant'

for directory in glob.glob(path):
    if os.path.isdir(directory):
        os.makedirs(f'{directory}/lables')
        coco_file = f'{directory}/coco_json.json'
        source_image_dir = f'{directory}/images'
        batch_pred(coco_file, source_image_dir)






