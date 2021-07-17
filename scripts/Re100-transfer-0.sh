#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/transfer/Re100to100-1s.yaml \
--start 0 \
--stop 10 \
--log;
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/transfer/Re200to100-1s.yaml \
--start 0 \
--stop 10 \
--log;