#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3 run_pino3d.py \
--config_path configs/scratch/Re500-scratch-05s.yaml \
--start 0 \
--stop 10 \
--log