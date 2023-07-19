#! /bin/bash
for i in {0..49}
do
CUDA_VISIBLE_DEVICES=1 python3 instance_opt.py --config configs/instance/Re500-1_8-PINO.yaml --ckpt checkpoints/Re500-1_8s-800-PINO-140000.pt --idx $i --tqdm
done