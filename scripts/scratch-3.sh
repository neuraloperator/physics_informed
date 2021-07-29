#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/scratch/Re250-scratch-1s.yaml \
--start 80 \
--stop 100 \
--log