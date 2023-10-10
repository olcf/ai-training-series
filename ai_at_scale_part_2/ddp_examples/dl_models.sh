#!/bin/bash

# Load Environment
source /lustre/orion/world-shared/stf218/sajal/miniconda3/bin/activate
conda activate /lustre/orion/world-shared/stf218/sajal/TORCH2/env-py310-rccl
export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"

# Downlaod gpt-j-6b model and tokenizer
python dl_models.py "EleutherAI/gpt-j-6b"

# Download gpt2-xl model and tokienizer
python dl_models.py "gpt2-xl"
