#!/bin/bash

# Fail on error
set -e

# Initialize conda (important for non-interactive shells)
source ~/anaconda3/etc/profile.d/conda.sh  # adjust path if needed

# Activate environment
conda activate afpi

# Run experiment (log both stdout and stderr)
PYTHONWARNINGS=ignore \
python skip_inv.py \
  --K_round 500 \
  --num_of_ddim_steps 50 \
  --delta_threshold 5e-13 \
  --guidance_scale 7 \
  --output outputs/test \
  --seed 0 \
  2>&1 | tee run.log