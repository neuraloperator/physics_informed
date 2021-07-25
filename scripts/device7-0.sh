#! /bin/bash
CUDA_VISIBLE_DEVICES=7 python3 run_pino3d.py \
--config_path configs/transfer/Re500to100-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=7 python3 run_pino3d.py \
--config_path configs/transfer/Re500to200-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=7 python3 run_pino3d.py \
--config_path configs/transfer/Re500to250-1s.yaml \
--start 0 \
--stop 40 \
--log;