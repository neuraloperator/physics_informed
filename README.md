# PINO

## Requirements
- Pytorch 1.8.0 or later
- wandb
- tqdm

## Data description

## Training scripts
To train on Navier Stokes equation, use
```bash
python3 train_PINO3d.py --config_path configs/***.yaml 
```

Configuration file format: see .yaml files under folder `configs` for example.

To pretrain neural operator, use
```bash
python3 pretrain.py --config_path configs/pretrain/[configuration file name].yaml
```

To train Navier Stokes equations sequentially without running `train_PINO3d.py` multiple times, use

```bash
python3 run_pino3d.py --config_path configs/[configuration file name].yaml --start [index of the first data] --stop [which data to stop]
```

## Operator learning
### Train

### Evaluate error of operator
`python3 val_operator.py --config_path configs/validate/Re500-05s.yaml`