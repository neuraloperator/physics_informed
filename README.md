# PINO

## Requirements
- Pytorch 1.8.0 or later
- wandb
- tqdm

## Description
To train on Navier Stokes equations with Reynolds number 500, use
```bash
python3 train_PINO3d.py --config_path configs/***.yaml 
```

To train on Navier Stokes equations with Reynolds number 100, 200, 250, 300, 350, 400, add '--new' to switch data loader to read data.
```bash
python3 train_PINO3d.py --config_path configs/***.yaml --new
```