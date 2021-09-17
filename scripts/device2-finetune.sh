#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python3 train_PINO3d.py \
--config_path configs/finetune/Re500-finetune-05s4k4.yaml \
--start 0 \
--log;
CUDA_VISIBLE_DEVICES=2 python3 train_PINO3d.py \
--config_path configs/finetune/Re500-finetune-05s4k4.yaml \
--start 1 \
--log;
CUDA_VISIBLE_DEVICES=2 python3 train_PINO3d.py \
--config_path configs/finetune/Re500-finetune-05s4k4.yaml \
--start 2 \
--log;
CUDA_VISIBLE_DEVICES=2 python3 train_PINO3d.py \
--config_path configs/finetune/Re500-finetune-05s4k4.yaml \
--start 3 \
--log;
CUDA_VISIBLE_DEVICES=3 python3 train_PINO3d.py \
--config_path configs/finetune/Re500-finetune-05s4k4.yaml \
--start 4 \
--log;
CUDA_VISIBLE_DEVICES=3 python3 train_PINO3d.py \
--config_path configs/finetune/Re500-finetune-05s4k4.yaml \
--start 5 \
--log;
CUDA_VISIBLE_DEVICES=3 python3 train_PINO3d.py \
--config_path configs/finetune/Re500-finetune-05s4k4.yaml \
--start 6 \
--log;
CUDA_VISIBLE_DEVICES=3 python3 train_PINO3d.py \
--config_path configs/finetune/Re500-finetune-05s4k4.yaml \
--start 7 \
--log;

