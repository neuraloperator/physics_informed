#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3 train_PINO3d.py \
--config_path configs/finetune/Re500-finetune-05s4k.yaml \
--start 9 \
--log;


