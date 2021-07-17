#! /bin/bash
CUDA_VISIBLE_DEVICES=3 python3 run_pino3d.py \
--config_path configs/transfer/Re100to300-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=3 python3 run_pino3d.py \
--config_path configs/transfer/Re200to300-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=3 python3 run_pino3d.py \
--config_path configs/transfer/Re250to300-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=3 python3 run_pino3d.py \
--config_path configs/transfer/Re300to300-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=3 python3 run_pino3d.py \
--config_path configs/transfer/Re350to300-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=3 python3 run_pino3d.py \
--config_path configs/transfer/Re400to300-1s.yaml \
--start 0 \
--stop 40 \
--log;
