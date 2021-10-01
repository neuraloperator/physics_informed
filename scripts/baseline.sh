#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3 nsfnet.py \
--config_path configs/scratch/Re500-scratch-05s.yaml \
--start 0 \
--stop 1 \
--logfile log/pinns.csv
