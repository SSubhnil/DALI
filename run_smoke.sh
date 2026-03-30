#!/bin/bash
# Smoke test launcher for DALI
# Usage: ./run_smoke.sh <gpu_id> <config_preset> <logdir>
set -e

GPU_ID="${1:?Usage: run_smoke.sh <gpu_id> <config_preset> <logdir>}"
CONFIG="${2:?}"
LOGDIR="${3:?}"

export LD_LIBRARY_PATH="/opt/conda/lib:/home/jovyan/cwm-workspace/conda_envs/dali/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/opt/conda/lib/libOSMesa.so"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$GPU_ID"

PYTHON=/home/jovyan/cwm-workspace/conda_envs/dali/bin/python

cd /home/jovyan/cwm-workspace/projects/DALI

exec $PYTHON -m contextual_mbrl.dreamer.train \
  --configs carl $CONFIG \
  --run.steps 1500 --run.eval_initial False --run.eval_every 1e9 \
  --run.save_every 500 --run.log_every 100 \
  --envs.amount 1 \
  --logdir "$LOGDIR" \
  --wandb.project '' --seed 42
