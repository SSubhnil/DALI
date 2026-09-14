#!/usr/bin/env bash
# §8b gradual-shift battery for DALI checkpoints. GPU + hardware EGL; NOT for the CPU test box.
#
#   bash drift_eval/launch_drift_eval.sh <gpu_index> <bench> [seed ...]
#   benches: w3f w2fphys w2fmix quadruped_2f_phys quadruped_2f_mixed
#   env overrides: CONDITIONS="A B C D E" N_EPISODES=64 NUM_ENVS=5 DRIFT_OUT=<dir>
#                  CAUSAL_WORLD_MODEL_ROOT=<CWM tree>
#                  DRIFT_GENERATOR_PATH=<generator .py> (optional override of the vendored copy;
#                  its sha256 must match vendor/VENDORED.md or DRIFT_GENERATOR_SHA256)
#
# One process per checkpoint, sequential over the requested seeds (default: every seed listed for
# the bench in drift_eval.yaml). Wall-clock and VRAM are UNMEASURED: take them from the 1-episode
# GPU smoke first (CONDITIONS="A B" N_EPISODES=10 on one seed).
set -euo pipefail

GPU="${1:?usage: launch_drift_eval.sh <gpu_index> <bench> [seed ...]}"
BENCH="${2:?usage: launch_drift_eval.sh <gpu_index> <bench> [seed ...]}"
shift 2
SEEDS=("$@")

DALI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CAUSAL_WORLD_MODEL_ROOT="${CAUSAL_WORLD_MODEL_ROOT:-/home/jovyan/m10_code/45e8d5d}"
GEN="${DRIFT_GENERATOR_PATH:-${DALI_REPO}/drift_eval/vendor/drift_generator.py}"
CONDITIONS="${CONDITIONS:-A B C D E}"
N_EPISODES="${N_EPISODES:-64}"
NUM_ENVS="${NUM_ENVS:-5}"
OUT="${DRIFT_OUT:-${DALI_REPO}/drift_eval_results}"

# Rendering: hardware EGL only. Never OSMesa (TensorFlow SIGSEGVs under it). On the devbox the DALI
# venv's vendored glvnd loader is set up by the CWM script below; on a Kubeflow pod export the
# equivalent (MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl, a libEGL loader on LD_LIBRARY_PATH) before calling.
if [[ -f "${CAUSAL_WORLD_MODEL_ROOT}/benchmark/devbox_dali_env.sh" && -z "${KUBERNETES_SERVICE_HOST:-}" ]]; then
  # shellcheck disable=SC1091
  source "${CAUSAL_WORLD_MODEL_ROOT}/benchmark/devbox_dali_env.sh"
fi
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" EGL_DEVICE_ID="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export PYTHONPATH="${DALI_REPO}"
PY="${DALI_PYTHON:-/home/jovyan/envs/dali/bin/python}"

echo "[drift] bench=${BENCH} gpu=${GPU} generator=${GEN}"
echo "[drift] generator sha256 $(sha256sum "${GEN}" | cut -d' ' -f1)"

SCHED="${OUT}/schedules/${BENCH}"
if [[ ! -f "${SCHED}/GENERATOR_PROVENANCE.json" ]]; then
  "${PY}" -m drift_eval.schedule --bench "${BENCH}" --out-root "${OUT}/schedules" --episodes "${N_EPISODES}" \
      --generator "${GEN}"
fi

mapfile -t ROWS < <("${PY}" - "${DALI_REPO}/drift_eval/drift_eval.yaml" "${BENCH}" "${SEEDS[@]}" <<'EOF'
import os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
bench, want = sys.argv[2], {int(s) for s in sys.argv[3:]}
rows = cfg["checkpoints"][bench]
for r in rows:
    if want and r["seed"] not in want:
        continue
    if not os.path.exists(os.path.join(r["logdir"], "checkpoint.ckpt")):
        sys.exit(f"missing checkpoint: {r['logdir']}")
    print(r["logdir"])
EOF
)
[[ ${#ROWS[@]} -gt 0 ]] || { echo "no checkpoints selected"; exit 1; }

for LOGDIR in "${ROWS[@]}"; do
  echo "[drift] ${LOGDIR}"
  # shellcheck disable=SC2086
  "${PY}" -m drift_eval.run_drift_eval --logdir "${LOGDIR}" --bench "${BENCH}" \
      --schedule-dir "${SCHED}" --results-root "${OUT}" --conditions ${CONDITIONS} \
      --n-episodes "${N_EPISODES}" --num-envs "${NUM_ENVS}" \
      --generator "${GEN}"
done
