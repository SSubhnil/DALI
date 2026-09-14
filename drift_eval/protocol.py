"""§8b gradual-shift protocol constants for the DALI port — the ONE place units, benches, seeds,
condition ids and the CausalWorldModel (CWM) source files are defined.

Single sources (never re-implemented here):
  * schedule generator  — CWM `benchmark/drift_generator.py`, VENDORED byte-identical at
    `drift_eval/vendor/drift_generator.py` (provenance + sha256 in `vendor/VENDORED.md`); loading
    refuses any file whose sha256 differs; the loaded sha256 is recorded in every output file.
  * factor value tables — CWM `config_files/latent_factor_values.yaml`.
  * regime-B wrappers   — CWM `benchmark/wrappers/carl_intra_episode.py` / `carl_reward_switch.py`
    (the modules DALI's own `contextual_mbrl/dreamer/envs.py` imports).
  * trace sink / CRN plan / trace-cell writer — CWM `envs/schedule_channel.py`,
    `benchmark/switch_recovery.py`.
The CWM root is `CAUSAL_WORLD_MODEL_ROOT` (the variable DALI's envs.py already reads); this module
defaults it to the pinned clean tree below when unset.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys

PINNED_CWM_ROOT = "/home/jovyan/m10_code/45e8d5d"
PINNED_CWM_COMMIT = "45e8d5d"
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
VENDORED_GENERATOR = os.path.join(VENDOR_DIR, "drift_generator.py")
VENDORED_MD = os.path.join(VENDOR_DIR, "VENDORED.md")


def vendored_sha256() -> str:
    with open(VENDORED_MD) as f:
        for line in f:
            item = line.strip().lstrip("-").strip()
            if item.startswith("sha256:"):
                return item.split(":", 1)[1].strip()
    raise RuntimeError(f"{VENDORED_MD} records no sha256")


def cwm_root() -> str:
    root = os.environ.get("CAUSAL_WORLD_MODEL_ROOT") or PINNED_CWM_ROOT
    os.environ["CAUSAL_WORLD_MODEL_ROOT"] = root
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def generator_path() -> str:
    """Default: the vendored copy. Optional override: DRIFT_GENERATOR_PATH (e.g. the devbox worktree
    $CAUSAL_WORLD_MODEL_ROOT/benchmark/drift_generator.py at befabde or later)."""
    return os.environ.get("DRIFT_GENERATOR_PATH") or VENDORED_GENERATOR


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_generator(path: str | None = None, expected_sha256: str | None = None):
    """Import the generator from `path` and return (module, path, sha256).

    The file's sha256 must equal `expected_sha256`, else DRIFT_GENERATOR_SHA256, else the hash
    recorded in vendor/VENDORED.md, so an override is either byte-identical to the vendored copy or
    carries its own explicitly recorded hash. The generator reads its bench presets relative to its
    own repo root (`_REPO_ROOT`); a vendored copy has none, so it is pointed at the CWM root."""
    path = os.path.abspath(path or generator_path())
    sha = sha256_file(path)
    want = expected_sha256 or os.environ.get("DRIFT_GENERATOR_SHA256") or vendored_sha256()
    if sha != want:
        raise RuntimeError(f"[drift_eval] generator {path} has sha256 {sha}, expected {want}; "
                           f"refusing to load it")
    spec = importlib.util.spec_from_file_location("_drift_generator_loaded", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "_REPO_ROOT"):
        mod._REPO_ROOT = cwm_root()
    return mod, path, sha


def values_yaml_path() -> str:
    return os.path.join(cwm_root(), "config_files", "latent_factor_values.yaml")


def load_values():
    import yaml
    with open(values_yaml_path()) as f:
        return yaml.safe_load(f)


CWM_PRESET_YAMLS = {  # output field -> path under CAUSAL_WORLD_MODEL_ROOT
    "cwm_sha256_experiments_dmc_yaml": "config_files/experiments/dmc.yaml",
    "cwm_sha256_configure_yaml": "config_files/configure.yaml",
    "cwm_sha256_latent_factor_values_yaml": "config_files/latent_factor_values.yaml",
}


def cwm_git_state(root: str | None = None):
    """(commit, dirty_count) of the CWM root; ("not a git checkout", "not a git checkout") if it is
    not one. `--no-optional-locks` keeps `git status` from writing the checkout's index."""
    import subprocess
    root = root or cwm_root()
    try:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "--no-optional-locks", "-C", root, "status",
                                          "--porcelain"], stderr=subprocess.DEVNULL).decode()
        return commit, sum(1 for line in status.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not a git checkout", "not a git checkout"


def cwm_provenance() -> dict:
    """The CWM fields every output file records: root, commit, dirty count, preset yaml sha256s."""
    root = cwm_root()
    commit, dirty = cwm_git_state(root)
    out = {"cwm_root": root, "cwm_commit": commit, "cwm_dirty_count": dirty}
    for field, rel in CWM_PRESET_YAMLS.items():
        out[field] = sha256_file(os.path.join(root, rel))
    return out


def source_provenance(generator_file: str | None = None) -> dict:
    """sha256 of every CWM file the port reads protocol or behaviour from."""
    root = cwm_root()
    rel = ["config_files/latent_factor_values.yaml",
           "config_files/experiments/dmc.yaml",
           "config_files/configure.yaml",
           "benchmark/wrappers/carl_intra_episode.py",
           "benchmark/wrappers/carl_reward_switch.py",
           "envs/dmc_reward_factors.py",
           "envs/schedule_channel.py",
           "benchmark/switch_recovery.py"]
    out = dict(cwm_provenance())
    for r in rel:
        p = os.path.join(root, r)
        out[r] = sha256_file(p) if os.path.exists(p) else None
    g = os.path.abspath(generator_file or generator_path())
    out["generator_path"] = g
    out["generator_sha256"] = sha256_file(g)
    return out


# ── Units: the ONE gravity conversion ────────────────────────────────────────────────────────────
# Our factor tables and schedules carry gravity as a MULTIPLIER of 9.81 m/s² (CWM
# envs/dmc_latent_factors.py:266 sets opt.gravity[2] = value * (-9.81)). DALI's CARL wrapper takes
# ABSOLUTE m/s² (carl_intra_episode.py:347 sets opt.gravity[2] = -value). This is the only place
# either direction of that conversion is written.
GRAVITY_MS2_PER_MULTIPLIER = 9.81

# Canonical factor id <-> CARL feature name (the id IS the CWM factor index; joint_damping is the
# quadruped alias of id 0, not a fifth factor — dmc_latent_factors.py:14-17).
CARL_NAME_TO_FACTOR_ID = {"actuator_strength": 0, "joint_damping": 0, "gravity": 1,
                          "friction": 2, "wind_x": 3}
ADDITIVE_FACTOR_IDS = {3}


def canonical_to_carl(carl_name: str, value: float) -> float:
    """Our-unit value -> the unit DALI's CARL wrapper applies."""
    if carl_name == "gravity":
        return float(value) * GRAVITY_MS2_PER_MULTIPLIER
    return float(value)


def carl_to_canonical(carl_name: str, value: float) -> float:
    if carl_name == "gravity":
        return float(value) / GRAVITY_MS2_PER_MULTIPLIER
    return float(value)


def u_coordinate(factor_id: int, value: float) -> float:
    """The generator's natural coordinate (drift_generator.u_of): log for multiplicative factors."""
    return float(value) if factor_id in ADDITIVE_FACTOR_IDS else math.log(float(value))


# ── Clip under a SCHEDULE: CWM's applier, exactly ─────────────────────────────────────────────────
# CWM envs/dmc_latent_factors.py:_apply_factor (45e8d5d :257-274) clips IN OUR UNITS, then converts:
#   factor 0/1/2: value = float(np.clip(value, 0.1, 10.0));  gravity: opt.gravity[2] = value * (-9.81)
#   factor 3    : value = float(np.clip(value, -wind_clip, wind_clip)), wind_clip = max|v| over the
#                 four splits of the active wind table (:183-193).
# The port applies the same clamp, in the same units, before the ONE conversion below, so the applied
# factor vector equals CWM's bit for bit (e.g. quadruped H-S joint_damping 10.000000000000002 -> 10.0).
# DALI's own wrapper clip (gravity [0.98, 39.24] m/s², envs carl_intra_episode.py) is BYPASSED under a
# schedule: its bound would cut the H-S/H-A gravity path (peak 5.126x = 50.28 m/s²).
def our_clip_bounds(carl_name: str, wind_table: dict | None = None):
    if carl_name in ("actuator_strength", "joint_damping", "gravity", "friction"):
        return (0.1, 10.0)
    if carl_name == "wind_x":
        vals = (list(wind_table["train_values"]) + list(wind_table["test_iid_values"])
                + list(wind_table["test_ood_mild_values"]) + list(wind_table["test_ood_hard_values"]))
        wind_clip = max(abs(v) for v in vals) if vals else 50.0
        return (-wind_clip, wind_clip)
    raise ValueError(f"no CWM clip for {carl_name!r}")


def our_clip(carl_name: str, value: float, wind_table: dict | None = None) -> float:
    """CWM's clamp in OUR units: float(np.clip(value, lo, hi))."""
    import numpy as np
    lo, hi = our_clip_bounds(carl_name, wind_table)
    return float(np.clip(value, lo, hi))


# ── Benches ─────────────────────────────────────────────────────────────────────────────────────
# `carl_factors`   : the CARL feature names DALI's regime-B factory for `regime_b_factors` manages.
# `generator_bench` : the generator --bench key (befabde resolves factors/values/λ̄/reward modes
#                     from CWM config_files/experiments/dmc.yaml); `generator_factors`/`domain`
#                     document what that preset activates.
# `values_keys`    : factor id -> latent_factor_values.yaml key our stack uses for that bench.
# Quadruped benches: a mirror of the walker design ruled by trajd-advisor 2026-09-14, pending PI
# ratification; NOT in ICLR_PAPER_PLAN §8b.
BENCHES = {
    "w3f": dict(generator_bench="walker_3f_phys", task="carl_dmc_walker", env="dm_walker_walk", regime_b_factors="K3",
                carl_factors=["actuator_strength", "gravity", "wind_x"],
                generator_factors="actuator_strength,gravity,wind", domain="walker",
                values_keys={0: "actuator_strength", 1: "gravity", 3: "wind_walker"},
                mixed=False),
    "w2fphys": dict(generator_bench="walker_2f_phys", task="carl_dmc_walker", env="dm_walker_walk", regime_b_factors="K2",
                    carl_factors=["actuator_strength", "gravity"],
                    generator_factors="actuator_strength,gravity", domain="walker",
                    values_keys={0: "actuator_strength", 1: "gravity"}, mixed=False),
    "w2fmix": dict(generator_bench="walker_2f_mixed", task="carl_dmc_walker", env="dm_walker_walk", regime_b_factors="K3_mixed",
                   carl_factors=["gravity"], generator_factors="gravity", domain="walker",
                   values_keys={1: "gravity"}, mixed=True),
    "quadruped_2f_phys": dict(generator_bench="quadruped_2f_phys", task="carl_dmc_quadruped", env="dm_quadruped_walk", regime_b_factors="K2",
                              carl_factors=["joint_damping", "gravity"],
                              generator_factors="joint_damping,gravity", domain="quadruped",
                              values_keys={0: "joint_damping", 1: "gravity"}, mixed=False),
    "quadruped_2f_mixed": dict(generator_bench="quadruped_2f_mixed", task="carl_dmc_quadruped", env="dm_quadruped_walk", regime_b_factors="K3_mixed",
                               carl_factors=["gravity"], generator_factors="gravity",
                               domain="quadruped", values_keys={1: "gravity"}, mixed=True),
}


def expected_processes(bench: str, values: dict) -> dict:
    """physics_fK -> train value list our generator must have been called with for this bench."""
    b = BENCHES[bench]
    return {f"physics_f{fid}": [float(v) for v in values[key]["train_values"]]
            for fid, key in sorted(b["values_keys"].items())}


# ── Conditions (§8c letters) -> OUR condition ids ─────────────────────────────────────────────────
# A = training protocol (Bernoulli), B = IR-A, C = IR-S, D = H-A, E = H-S, F = τ-dial. Ids and axes are
# CWM's own (CausalWorldModel @befabde benchmark/eval_conditions.yaml): A is `phys_train`
# (axis physics, split train: :28/:47/:77); the drift ids are `drift_ir_s`, `drift_ir_a`, `drift_h_s`,
# `drift_h_a`, `drift_tau_<k>` (axis drift: :143-163).
TAUS = (1, 8, 16, 64, 256)
CONDITIONS = {
    "A": dict(id="phys_train", axis="physics", generator=None, label="training protocol"),
    "B": dict(id="drift_ir_a", axis="drift", generator="ir_a", label="matched abrupt (IR-A)"),
    "C": dict(id="drift_ir_s", axis="drift", generator="ir_s", label="gradual sine (IR-S)"),
    "D": dict(id="drift_h_a", axis="drift", generator="h_a", label="hard matched abrupt (H-A)"),
    "E": dict(id="drift_h_s", axis="drift", generator="h_s", label="hard gradual sine (H-S)"),
}
for _tau in TAUS:
    CONDITIONS[f"F{_tau}"] = dict(id=f"drift_tau_{_tau}", axis="drift", generator=f"tau_{_tau}",
                                  label=f"tau-dial tau={_tau}")

# Envs per batch: CWM Evaluate.NumEnvs = 5 (eval_conditions.yaml smoke rule "a multiple of NumEnvs 5").
# The retained episodes are crn_episode_plan(base, n, 5): workers finish in lockstep and eval.py:778 (@befabde)
# stops at the n-th completion, so with n = 64 the 13th round keeps workers 0-3 and drops worker 4.
DEFAULT_NUM_ENVS = 5


def reward_mode_ids(domain: str) -> dict:
    """mode -> CWM reward_mode_id: the index in the SORTED domain registry
    (envs/dmc_reward_factors.py:359 `_all_mode_names = sorted(...)`, :429 `_mode_name_to_id`)."""
    cwm_root()
    from envs.dmc_reward_factors import DOMAIN_REWARD_REGISTRIES
    return {m: i for i, m in enumerate(sorted(DOMAIN_REWARD_REGISTRIES[domain]().keys()))}


REWARD_FIXED_FILE = "reward_fixed.json"   # befabde drift_generator.build_reward_fixed_schedule :352 (mixed benches)


def schedule_filename(generator_condition: str) -> str:
    """File name the generator CLI writes (drift_generator.main: {cond}.json / tau_{τ}.json)."""
    return f"{generator_condition}.json"


# ── Seeds (CRN) ─────────────────────────────────────────────────────────────────────────────────
def eval_base_seed(trained_seed: int) -> int:
    """Our battery's env seed: CWM eval.py:48 `BasicSettings.Seed * 1000 + 7`."""
    return int(trained_seed) * 1000 + 7


def worker_seed(base_seed: int, worker_index: int) -> int:
    """CWM schedule_channel.crn_episode_plan: worker i is seeded base_seed + i."""
    return int(base_seed) + int(worker_index)


def worker_task_rng(worker_seed_: int):
    """CWM's reset seed derivation, per (worker, ordinal): worker i's DMControl is built with
    seed = base + i (eval.py:108 `DMControl(..., seed=seed)`), which loads the dm_control task with
    `task_kwargs={'random': seed}` (envs/my_dmc.py:43-50), i.e. ONE RandomState(worker seed) for the
    worker's lifetime; `DMControl.reset` does not reseed it (my_dmc.py:88-91), so episode j draws the
    j-th initial state from that stream. DALI's CARL env builds its task with random=None and rebuilds
    it on context changes; the port hands this one RNG to every rebuilt task."""
    import numpy as np
    return np.random.RandomState(int(worker_seed_))


EPISODE_T = 500  # decisions per episode at dmc.repeat 2 (1000 control steps)
