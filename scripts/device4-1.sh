#! /bin/bash
CUDA_VISIBLE_DEVICES=4 python3 run_pino3d.py \
--config_path configs/transfer/Re300to400-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=4 python3 run_pino3d.py \
--config_path configs/transfer/Re350to400-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=4 python3 run_pino3d.py \
--config_path configs/transfer/Re400to400-1s.yaml \
--start 0 \
--stop 40 \
--log;
