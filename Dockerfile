FROM nvcr.io/nvidia/pytorch:22.09-py3
RUN useradd -ms /bin/bash pino
USER pino
ENV PATH=/home/pino/.local/bin:$PATH
RUN pip install --user \
    wandb tqdm pyyaml