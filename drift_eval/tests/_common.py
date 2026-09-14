"""Shared CPU-only fixtures. No GPU, no GL: every test runs at the gymnasium level (MuJoCo physics,
no RenderImage). Run from the DALI repo root with the DALI venv:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=$PWD \
        /home/jovyan/envs/dali/bin/python -m unittest discover -s drift_eval/tests -t .
"""
import os
import sys
import tempfile
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MUJOCO_GL"] = "disable"          # physics only; no GL backend is ever initialised
os.environ.pop("PYOPENGL_PLATFORM", None)
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from drift_eval import protocol as P  # noqa: E402

P.cwm_root()

REWARD_MODES = {"carl_dmc_walker": ["walk_forward", "walk_backward"],
                "carl_dmc_quadruped": ["go_north", "go_south"]}
N_EPISODES = 4
BASE_SEED = 3007
_SCHED_ROOT = None
_GEN = None


def bench_config(bench, seed=3):
    b = P.BENCHES[bench]
    return {"task": b["task"], "seed": seed,
            "env": {"carl": {"context": "benchmark_discrete", "regime": "B",
                             "regime_b_factors": b["regime_b_factors"],
                             "regime_b_lambda": 0.005, "regime_b_lambda_range": [0.003, 0.008],
                             "regime_b_split": "train", "regime_b_factor_splits": [""],
                             "regime_b_reward_modes": REWARD_MODES[b["task"]]}}}


def generator():
    global _GEN
    if _GEN is None:
        _GEN = P.load_generator()
    return _GEN


def schedule_dir(bench):
    global _SCHED_ROOT
    from drift_eval import schedule as S
    if _SCHED_ROOT is None:
        _SCHED_ROOT = tempfile.mkdtemp(prefix="drift_eval_sched_")
    d = os.path.join(_SCHED_ROOT, bench)
    if not os.path.isdir(d):
        S.make_schedules(bench, _SCHED_ROOT, episodes=N_EPISODES)
    return d


def load_payload(bench, gen_cond):
    from drift_eval import schedule as S
    return S.load_schedule(os.path.join(schedule_dir(bench), f"{gen_cond}.json"), bench, gen_cond,
                           P.load_values(), generator()[0].GENERATOR_VERSION)


def load_reward_payload(bench):
    from drift_eval import schedule as S
    if not P.BENCHES[bench]["mixed"]:
        return None
    return S.load_reward_fixed(os.path.join(schedule_dir(bench), P.REWARD_FIXED_FILE),
                               REWARD_MODES[P.BENCHES[bench]["task"]], N_EPISODES,
                               generator()[0].GENERATOR_VERSION)


def zero_action(env):
    return np.zeros(env.action_space.shape, dtype=np.float32)
