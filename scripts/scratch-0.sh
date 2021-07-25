#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/scratch/Re200-scratch-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/scratch/Re250-scratch-1s.yaml \
--start 0 \
--stop 40 \
--log