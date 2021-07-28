#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/transfer/Re400to250-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/transfer/Re400to350-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/transfer/Re400to400-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/transfer/Re400to500-1s.yaml \
--start 20 \
--stop 40 \
--log

