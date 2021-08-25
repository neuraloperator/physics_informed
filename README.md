# PINO

## Requirements
- Pytorch 1.8.0 or later
- wandb
- tqdm

## Description
To train on Navier Stokes equations, use
```bash
python3 train_PINO3d.py --config_path configs/***.yaml --num_gpus 1 
```
see .yaml files under folder `configs` for example. 