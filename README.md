# PINO

## Requirements
- Pytorch 1.8.0 or later
- wandb
- tqdm
- scipy
- h5py
- numpy
- DeepXDE:latest
- tensorflow 2.4.0

## Data description
### Burgers equation
`burgers_pino.mat`

### Darcy flow 
- spatial domain: $x\in (0,1)^2$
- Data file: `piececonst_r421_N1024_smooth1.mat`, `piececonst_r421_N1024_smooth2.mat`
- Raw data shape: 1024x421x421


### Long roll out of Navier Stokes equation
- spatial domain: $x\in (0, 1)^2$
- temporal domain: $t\in \[0, 49\]$
- forcing: $0.1(\sin(2\pi(x_1+x_2)) + \cos(2\pi(x_1+x_2)))$
- viscosity = 0.001

Data file: `nv_V1e-3_N5000_T50.mat`, with shape 50 x 64 x 64 x 5000 

- train set: -1-4799
- test set: 4799-4999
### Navier Stokes with Reynolds number 500
- spatial domain: $x\in (0, 2\pi)^2$
- temporal domain: $t \in \[0, 0.5\]$
- forcing: $-4\cos(4x_2)$
- Reynolds number: 500

Train set: 

Test set: `NS_Re500_s256_T100_test.npy`

Configuration file format: see `.yaml` files under folder `configs` for detail. 

## Code for Burgers equation
### Train PINO
To run PINO for Burgers equation, use, e.g.,
```bash 
python3 train_burgers.py --config_path configs/pretrain/burger
```

To test PINO for burgers equation, use, e.g., 
```bash
python3 train_burgers.py --config_path configs/
```

## Code for Darcy Flow

### Operator learning
To run PINO for Darcy Flow, use, e.g., 
```bash
python3 train_operator.py --config_path configs/pretrain/Darcy-pretrain.yaml
```
To evaluate operator for Darcy Flow, use, e.g., 
```bash
python3 eval_operator.py --config_path configs/test/darcy.yaml
```

### Test-time optimization
To do test-time optimization for Darcy Flow, use, e.g., 
```bash
python3 run_pino2d.py --config_path configs/finetune/Darcy-finetune.yaml --start [starting index] --stop [stopping index]
```

### Baseline
To run DeepONet, use, e.g., 
```bash
python3 deeponet.py --config_path configs/pretrain/Darcy-pretrain-deeponet.yaml --mode train 
```
To test DeepONet, use, e.g., 
```bash
python3 deeponet.py --config_path configs/test/darcy.yaml --mode test
```


## Code for Navier Stokes equation
### Train PINO for short time period
To run operator learning, use, e.g., 
```bash
python3 train_operator.py --config_path configs/pretrain/Re500-pretrain-05s-4C0.yaml
```
To evaluate trained operator, use
```bash
python3 eval_operator.py --config_path configs/test/Re500-05s.yaml
```
To run test-time optimization, use
```bash
python3 train_PINO3d.py --config_path configs/***.yaml 
```

To train Navier Stokes equations sequentially without running `train_PINO3d.py` multiple times, use

```bash
python3 run_pino3d.py --config_path configs/[configuration file name].yaml --start [index of the first data] --stop [which data to stop]
```


### Baseline for short time period
To train DeepONet, use 
```bash
python3 deeponet.py --config_path configs/[configuration file].yaml --mode train
```

To test DeepONet, use 
```bash
python3 deeponet.py --config_path configs/[configuration file].yaml --mode test
```

To train and test PINNs, use, e.g.,  
```bash
python3 nsfnet.py --config_path configs/Re500-pinns-05s.yaml --start [starting index] --stop [stopping index]
```
### Baseline for long roll out
To train and test PINNs, use
```bash
python3 nsfnet.py --config_path configs/scratch/NS-50s.yaml --long --start [starting index] --stop [stopping index]
```

### Pseudospectral solver for Navier Stokes equation
To run solver, use 
```bash
python3 run_solver.py --config_path configs/[configuration file name].yaml
```
