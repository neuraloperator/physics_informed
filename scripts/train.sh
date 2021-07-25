#! /bin/bash
wandb login 69a3bddb4146cf76113885de5af84c7f4c165753;

bash scripts/device0.sh &
bash scripts/device1.sh &
bash scripts/device2.sh &
bash scripts/device3.sh &
bash scripts/device4.sh &
bash scripts/device5.sh &
bash scripts/device6.sh &
bash scripts/device7.sh
