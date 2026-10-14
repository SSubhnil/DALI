"""CPU-only helpers for env-level tests: no GPU, no GL (MuJoCo render stubbed to zeros).

Run from the DALI repo root with the DALI venv and a CausalWorldModel checkout:

    CUDA_VISIBLE_DEVICES= CAUSAL_WORLD_MODEL_ROOT=/path/to/CausalWorldModel PYTHONPATH=$PWD \
        /home/jovyan/envs/dali/bin/python -m unittest discover -s contextual_mbrl/tests -t .
"""
import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MUJOCO_GL"] = "disable"
os.environ.pop("PYOPENGL_PLATFORM", None)
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "dreamerv3_compat")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import ruamel.yaml as _yaml  # noqa: E402
from dm_control.mujoco import engine as _engine  # noqa: E402

_engine.Physics.render = (lambda self, height=240, width=320, *a, **kw:
                          np.zeros((height, width, 3), np.uint8))

from dreamerv3 import embodied  # noqa: E402

# DALI presets of the regime-B benches (configs.yaml).
BENCHES = {
    "walker_2f": "benchmark_regime_b",
    "walker_3f": "benchmark_regime_b_K3",
    "walker_mixed": "benchmark_regime_b_mixed",
    "quad_2f": "benchmark_regime_b_quadruped",
    "quad_mixed": "benchmark_regime_b_mixed_quadruped",
}


def bench_config(bench, seed=11, amount=4, **updates):
    master = _yaml.YAML(typ="safe").load(
        open(os.path.join(REPO, "contextual_mbrl", "dreamer", "configs.yaml")))
    cfg = embodied.Config(master["defaults"])
    for name in ("carl", BENCHES[bench], "enc_img_dec_img_ctxencoder_gru_grssm"):
        cfg = cfg.update(master[name])
    return cfg.update({"seed": seed, "envs.amount": amount, "envs.parallel": "none",
                       "envs.restart": False, "jax.platform": "cpu", **updates})


def chain(env):
    out, e = [], env
    for _ in range(64):
        out.append(e)
        e = getattr(e, "env", None) or getattr(e, "_env", None)
        if e is None:
            break
    return out


def find(env, cls_name):
    return next(x for x in chain(env) if type(x).__name__ == cls_name)


def rollout(env, steps, seed=0):
    """Step one embodied env `steps` decisions with random actions (episode resets via
    is_last). Returns per-step physics factor values of CARLIntraEpisodeSwitching, the
    reward mode in force (None without CARLRewardModeSwitcher) and obs['context']."""
    phys = find(env, "CARLIntraEpisodeSwitching")
    try:
        rew = find(env, "CARLRewardModeSwitcher")
    except StopIteration:
        rew = None
    rng = np.random.RandomState(seed)
    space = env.act_space["action"]
    act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
    act["reset"] = True
    factors, modes, contexts = [], [], []
    for _ in range(steps):
        act["action"] = rng.uniform(-1, 1, space.shape).astype(np.float32)
        obs = env.step(act)
        act["reset"] = bool(obs["is_last"])
        factors.append(tuple(np.round(np.asarray(phys._factor_values, np.float64), 6)))
        modes.append(None if rew is None else rew._current_mode)
        contexts.append(np.asarray(obs["context"], np.float32).copy() if "context" in obs else None)
    return factors, modes, contexts
