#!/bin/bash
python3 track.py --source 0 --yolo_model yolov5_v7/weights/best_388.pt --name suf_testing --imgsz 3840 --device 0 --conf-thres 0.1 --use-bytetracker --classes 0  --tracker-lowfps 16 --magi-alg
