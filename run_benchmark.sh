#!/bin/bash
# DALI benchmark runner with OSMesa from base conda
# Avoids LLVM conflicts by pre-loading OSMesa before JAX/TF

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled

PYTHON=/home/jovyan/miniconda3/envs/dali/bin/python

exec $PYTHON "$@"
