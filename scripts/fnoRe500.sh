#! /bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres gpu:v100:1
#SBATCH --mem=64G
#SBATCH --email-user=hzzheng@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

python3 train_operator.py --config_path configs/operator/Re500-FNO.yaml