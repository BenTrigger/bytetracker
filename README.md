# Baloons


## Getting started

pip install -r requirements.txt

if not finding extra labs you can find them in here:
http://mthi-sparx.maman.iai:8082/teama_i/baloons/-/tree/main/ByteTrack/packages_to_install
pip install them

## Params:
--source    path to data(files or folders with videos or images) or for camera just input 0

--use-bytetracker

--tracker-lowfps 32         (32 for fps 1-6,  1-4 for fps 7-15,  1 for fps 16+)

--add-yolobboxes    no need- only for debugging

--deep_sort_model custom_data_ckpt_epoch_1000

--conf-thres 0.24  you can change it

--agnostic-nms  if you dont want to find overlap objects

--name  name of the running test

--yolo_model path_to_weights

--device 0   (gpu 0)

--conf-thres 0.24   (minimum confidence 24% for yolo detection)

--imgsz 3840

--save-txt  save labels

--save-vid : saving video picture, if you ADD --out-video so it will save video.

--save-txt : saving labels of everything(video as 1 file, camera for each frame it will get label)


full example:
```
python3.8 track.py --source yolov5ben\data\images --name tracking_medolal_all_lowfps32 --yolo_model weights/baloons_sliced_1600_train_all_frames_batch4_250epochs_no_birds_augmentaions_union_dronelight/best.pt --agnostic-nms --save-txt --device 0 --imgsz 3840 --conf-thres 0.24 --use-bytetracker --deep_sort_model custom_data_ckpt_epoch_1000 --tracker-lowfps 32
```

## Yolo Weights: best.py
- [ ]    param:
    --yolo_model weights/baloons_sliced_1600_train_all_frames_batch4_250epochs_no_birds_augmentaions_union_dronelight/best.py

    git path:
- [ ] http://mthi-sparx.maman.iai:8082/teama_i/baloons/-/tree/main/weights/baloons_sliced_1600_train_all_frames_batch4_250epochs_no_birds_augmentaions_union_dronelight




## inference- run in on server cases:

BYTE_TRACKER:
```
(SAVE IMAGES AND LABELS for images)
python3.8 track.py --source data/test/ --yolo_model path_to_best.pt --name testing_byte --imgsz 3840 --tracker-lowfps 32 --save-txt --device 0 --conf-thres 0.1 --save-vid --use-bytetracker
```
```
(SAVE VIDEO AND LABELS for images)
python3.8 track.py --source data/test/ --yolo_model path_to_best.pt --name testing_byte --imgsz 3840 --tracker-lowfps 32 --save-txt --device 0 --conf-thres 0.1 --save-vid --out-video --use-bytetracker
```
```
(SAVE VIDEO AND LABELS for VIDEO)
python3.8 track.py --source data/detect_for_erik/black_panther_04-06-2021_10_42-10_43_cam_0.avi --yolo_model path_to_best.pt --tracker-lowfps 32 --name testing_byte --imgsz 3840 --save-txt --device 0 --conf-thres 0.1 --save-vid --out-video --use-bytetracker
```
```
(CAMERA SAVE RESULTS: FRAME ANY AND LABEL FOR EACH DETECT)
python3.8 track.py --source 0 --yolo_model path_to_best.pt --name testing_byte --imgsz 3840 --tracker-lowfps 32 --save-txt --device 0 --conf-thres 0.1 --use-bytetracker --save-vid --save-txt  # NO NEED OUT-VIDEO IN HERE
```
```
(CAMERA)
python3.8 track.py --source 0 --yolo_model path_to_best.pt --name testing_byte --imgsz 3840 --tracker-lowfps 32 --save-txt --device 0 --conf-thres 0.1 --out-video --use-bytetracker  # --save-vid
```

--save-vid : saving video picture, if you ADD --out-video so it will save video.
--save-txt : saving labels of everything(video as 1 file, camera for each frame it will get label)
