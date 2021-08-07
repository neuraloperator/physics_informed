#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/transfer/Re400to500-1s.yaml \
--start 0 \
--stop 80 \
--log;



