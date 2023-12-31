#!/bin/bash

set +e

# inference on 4 MOT16 video sequences at the same time
# suits a 4GB GRAM GPU, feel free to increase if you have more memory
N=1

# generate tracking results for each sequence
for i in MOT16-02 MOT16-04 MOT16-05 MOT16-09 MOT16-10 MOT16-11 MOT16-13
do
	(
		# change name to inference source so that each thread write to its own .txt file
		if [ ! -d ~/Yolov5_DeepSort_Pytorch/MOT16_eval/TrackEval/data/MOT16/train/$i/$i ]
		then
			mv ~/Yolov5_DeepSort_Pytorch/MOT16_eval/TrackEval/data/MOT16/train/$i/img1/ ~/Yolov5_DeepSort_Pytorch/MOT16_eval/TrackEval/data/MOT16/train/$i/$i
		fi
		# run inference on sequence frames
		python3 track.py --source ~/Yolov5_DeepSort_Pytorch/MOT16_eval/TrackEval/data/MOT16/train/$i/$i --save-txt --evaluate --yolo_model yolov5/weights/crowdhuman_yolov5m.pt --classes 0 --exist-ok
	    # move generated results to evaluation repo
	) &
	# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
	# allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]
	then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi
done

# no more jobs to be started but wait for pending jobs
# (all need to be finished)
wait
echo "Inference on all MOT16 sequences DONE"

echo "Moving data from experiment folder to MOT16"
mv ~/Yolov5_DeepSort_Pytorch/runs/track/exp/* \
   ~/Yolov5_DeepSort_Pytorch/MOT16_eval/TrackEval/data/trackers/mot_challenge/MOT16-train/ch_yolov5m_deep_sort/data/

# run the evaluation
python ~/Yolov5_DeepSort_Pytorch/MOT16_eval/TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 \
 --TRACKERS_TO_EVAL ch_yolov5m_deep_sort --SPLIT_TO_EVAL train --METRICS CLEAR Identity \
 --USE_PARALLEL False --NUM_PARALLEL_CORES 4
