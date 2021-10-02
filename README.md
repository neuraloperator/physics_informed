# PINO

## Requirements
- Pytorch 1.8.0 or later
- wandb
- tqdm

## Data description

## Navier Stokes equation
### Train PINO
To run operator learning, use
```bash
python3 train_operator.py --config_path configs/pretrain/[configuration file name].yaml
```
To evaluate trained operator, use
```bash
python3 val_operator.py --config_path configs/validate/Re500-05s.yaml
```
To run test-time optimization, use
```bash
python3 train_PINO3d.py --config_path configs/***.yaml 
```

To train Navier Stokes equations sequentially without running `train_PINO3d.py` multiple times, use

```bash
python3 run_pino3d.py --config_path configs/[configuration file name].yaml --start [index of the first data] --stop [which data to stop]
```


### Baseline
To train DeepONet, use 
```bash
python3 deeponet_ns.py --config_path configs/[configuration file].yaml --mode train
```

To test DeepONet, use 
```bash
python3 deeponet_ns.py --config_path configs/[configuration file].yaml --mode test
```

To train and test PINNs, use 
```bash
python3 nsfnet.py --config_path configs/[configuration name].yaml 
--logfile [log file path] --start [starting index] --stop [stopping index]
```


Configuration file format: see .yaml files under folder `configs` for example.
