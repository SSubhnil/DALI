"""DALI env chain for the §8b battery, with the schedule carried BY THE ENV, never by a global.

The defect this module must not reproduce (CWM plan §8b, fixed in CWM 1a30c23): an eval runner
installed the schedule as a parent-process module global, the stepping process was spawned, and
every "scheduled" episode silently ran unscheduled. The DALI-side wrappers have the same global
(`carl_intra_episode._ACTIVE_FACTOR_SCHEDULE`). Here, as in CWM `envs/schedule_channel.py`:

  * the payload is an explicit constructor argument of the wrapper instance, so it lives in
    whichever process builds (and therefore steps) the env, and `gym_env_fn` is a picklable
    partial that carries it across a spawn;
  * `mode='scheduled'` with no payload RAISES at construction (fail closed);
  * after every reset and every step the wrapper checks, in the stepping process, that its own
    factor state AND the MuJoCo model equal the scheduled value exactly, and raises otherwise;
  * per-episode traces go to a directory (CWM `DirTraceSink`), since a list cannot cross a spawn.

Env parity with CWM `envs/dmc_latent_factors.py` (verifier PARITY_TABLE.md, 2026-09-14):
  * timing: CWM writes the factor and then steps with NO mj_forward (:440/:451); the CARL wrapper calls
    `physics.forward()` after a switch (carl_intra_episode.py:630-631), so it is suppressed during step;
  * draw order: CWM draws per factor in factor-id order (:367/:407); CARL sorts names alphabetically
    (quadruped: gravity before joint_damping), so the names are re-sorted by factor id;
  * value tables: CWM's multipliers, gravity computed as CWM does (0.4 × 9.81), not CARL's stored 3.924;
  * reward RNG: seeded with the worker seed, as CWM eval.py:118 seeds DMControlRewardFactors.

The chain mirrors DALI `contextual_mbrl/dreamer/envs.py:822-963` (create_wrapped_carl_env, regime B)
and swaps DALI's regime-B physics wrapper / reward switcher for subclasses built from the SAME
factory output (same factor configs, value sets, λ range, action repeat, seed).
"""
from __future__ import annotations

import dataclasses
import functools
import itertools
import os

import numpy as np

from drift_eval import protocol as P

P.cwm_root()

from benchmark.wrappers import carl_intra_episode as _cie  # noqa: E402
from benchmark.wrappers.carl_reward_switch import CARLRewardModeSwitcher  # noqa: E402

MODES = ("bernoulli", "scheduled")

_FACTORIES = {
    ("dmc_walker", "K2"): _cie.make_walker_regime_b_wrapper,
    ("dmc_walker", "K3"): _cie.make_walker_regime_b_K3_wrapper,
    ("dmc_walker", "K3_mixed"): _cie.make_walker_regime_b_mixed_wrapper,
    ("dmc_quadruped", "K2"): _cie.make_quadruped_regime_b_wrapper,
    ("dmc_quadruped", "K3"): _cie.make_quadruped_regime_b_K3_wrapper,
    ("dmc_quadruped", "K3_mixed"): _cie.make_quadruped_regime_b_mixed_wrapper,
}


class _EpisodeRecorder:
    """Per-step rows in the CWM D1 worker-trace schema (dmc_latent_factors.py:319-330)."""

    def __init__(self, trace_dir, worker_tag):
        self.trace_dir, self.worker_tag = trace_dir, worker_tag
        self.sink = None
        self.in_memory = []          # finished episodes when no trace_dir (tests)
        self.rows = None

    def start(self):
        self.rows = {"reward": [], "reward_mode_id": [], "factor_values": [], "decision_step": []}

    def append(self, reward, mode_id, canon_row, decision_step):
        self.rows["reward"].append(float(reward))
        self.rows["reward_mode_id"].append(int(mode_id))
        self.rows["factor_values"].append(np.asarray(canon_row, dtype=np.float32))
        self.rows["decision_step"].append(int(decision_step))

    def flush(self, factor_active, extra):
        r = self.rows
        trace = {"reward": np.asarray(r["reward"], dtype=np.float32),
                 "reward_mode_id": np.asarray(r["reward_mode_id"], dtype=np.int16),
                 "factor_values": np.stack(r["factor_values"]).astype(np.float32),
                 "factor_active": np.asarray(factor_active, dtype=bool),
                 "decision_step": np.asarray(r["decision_step"], dtype=np.int32),
                 **{k: np.asarray(v) for k, v in extra.items()}}
        if self.trace_dir is not None:
            if self.sink is None:
                from envs.schedule_channel import DirTraceSink
                self.sink = DirTraceSink(self.trace_dir, self.worker_tag)
            self.sink.append(trace)
        else:
            self.in_memory.append(trace)
        self.rows = None


def _no_forward(*args, **kwargs):
    return None


class ScheduledCARLSwitching(_cie.CARLIntraEpisodeSwitching):
    """DALI's regime-B physics wrapper with an instance-level schedule (or its native Bernoulli)."""

    def __init__(self, env, factor_configs, train_ranges, seed, initial_mode, action_repeat, *,
                 mode, schedule=None, worker_index=0, num_envs=1, trace_dir=None,
                 wind_table=None):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if mode == "scheduled" and schedule is None:
            raise RuntimeError(
                f"[drift_eval] mode='scheduled' but no schedule payload reached this process "
                f"(pid {os.getpid()}). A parent-process global does not cross a spawn; pass the "
                f"payload to the env constructor.")
        if mode == "bernoulli" and schedule is not None:
            raise RuntimeError("[drift_eval] mode='bernoulli' was given a schedule payload")
        super().__init__(env, factor_configs=factor_configs, train_ranges=train_ranges,
                         seed=seed, initial_mode=initial_mode, action_repeat=action_repeat)
        # Draw order = CWM factor-id order (dmc_latent_factors.py:367/:407), not CARL's sorted names.
        order = sorted(range(len(self._factor_names)),
                       key=lambda i: P.CARL_NAME_TO_FACTOR_ID[self._factor_names[i]])
        self._factor_names = [self._factor_names[i] for i in order]
        for attr in ("_train_low", "_train_high", "_current_lambdas", "_factor_values",
                     "_factor_changed", "_segment_ids", "_steps_since_switch"):
            setattr(self, attr, np.asarray(getattr(self, attr))[order])
        self._wind_factor_idx = (self._factor_names.index("wind_x")
                                 if "wind_x" in self._factor_names else None)
        self._drift_mode = mode
        self._drift_worker_index = int(worker_index)
        self._drift_worker_seed = int(seed)
        self._drift_num_envs = int(num_envs)
        self._drift_offset = int(seed)       # CWM dmc_latent_factors.py:221 `_sched_offset = seed`
        self._drift_schedule_hash = None if schedule is None else schedule.get("schedule_hash")
        self._external_recorder = False
        self._recorder = _EpisodeRecorder(trace_dir, f"w{self._drift_worker_index}lf")
        self._last_canonical = None
        self.reset_canonical = None
        self._carl = self._find_carl_env()
        self._install_task_seeding()
        if mode == "scheduled":
            self._load_payload(schedule, wind_table)

    @classmethod
    def from_native(cls, native, seed, **kw):
        train_ranges = {n: (float(native._train_low[i]), float(native._train_high[i]))
                        for i, n in enumerate(native._factor_names)}
        return cls(native.env, factor_configs=dict(native._factor_configs),
                   train_ranges=train_ranges, seed=seed, initial_mode=native._initial_mode,
                   action_repeat=native._action_repeat, **kw)

    def use_cwm_value_tables(self, values, values_keys, split, factor_splits):
        """Replace each managed factor's value set with CWM's table (latent_factor_values.yaml) in
        CARL units, gravity computed as CWM computes it (value * 9.81), not CARL's stored 3.924."""
        self._factor_configs = dict(self._factor_configs)
        for name in self._factor_names:
            key = values_keys[P.CARL_NAME_TO_FACTOR_ID[name]]
            sp = (factor_splits or {}).get(name, split)
            table = [P.canonical_to_carl(name, v) for v in values[key][f"{sp}_values"]]
            self._factor_configs[name] = dataclasses.replace(self._factor_configs[name],
                                                             values=table)

    # -- payload --------------------------------------------------------------------------------
    def _load_payload(self, schedule, wind_table):
        T = int(schedule["T"])
        expected = {f"physics_f{P.CARL_NAME_TO_FACTOR_ID[n]}": n for n in self._factor_names}
        self._sched_T = T
        self._sched_canon, self._sched_carl_events = [], []
        self._sched_clipped_entries = 0
        for j, ep in enumerate(schedule["episodes"]):
            if set(ep) != set(expected):
                raise RuntimeError(f"[drift_eval] schedule episode {j} keys {sorted(ep)} != "
                                   f"this wrapper's factors {sorted(expected)}")
            canon, events = {}, {}
            for key, name in expected.items():
                seq = ep[key]
                if [int(s) for s, _ in seq] != list(range(T + 1)):
                    raise RuntimeError(f"[drift_eval] schedule {key} is not dense 0..{T}")
                raw = np.array([float(v) for _, v in seq], dtype=np.float64)
                # CWM's clamp, in our units, BEFORE the unit conversion (protocol.our_clip).
                vals = np.array([P.our_clip(name, v, wind_table) for v in raw], dtype=np.float64)
                self._sched_clipped_entries += int((vals != raw).sum())
                canon[name] = vals
                events[key] = [[t, P.canonical_to_carl(name, v)] for t, v in enumerate(vals)]
            self._sched_canon.append(canon)
            self._sched_carl_events.append(events)
        # Our clamp is already applied; DALI's wrapper clip (native gravity bound 39.24 m/s²) is
        # bypassed so it cannot alter a scheduled value: np.clip(x, -inf, inf) == x.
        self._factor_configs = dict(self._factor_configs)
        for name in self._factor_names:
            self._factor_configs[name] = dataclasses.replace(self._factor_configs[name],
                                                             clip_range=(-np.inf, np.inf))

    def schedule_episode_index(self):
        return (self._drift_offset + self._episode_ordinal) % len(self._sched_carl_events)

    def _get_scheduled_episode(self):
        # Overrides the base lookup so the module global is never consulted.
        if self._drift_mode != "scheduled":
            return None
        return self._sched_carl_events[self.schedule_episode_index()]

    # -- CRN: seed the dm_control task RNG per episode ------------------------------------------
    def _find_carl_env(self):
        from carl.envs.carl_env import CARLEnv
        env = self.env
        for _ in range(10):
            if isinstance(env, CARLEnv):
                return env
            env = getattr(env, "env", None)
            if env is None:
                break
        raise RuntimeError("[drift_eval] no CARLEnv in the wrapped chain")

    def _install_task_seeding(self):
        carl, owner = self._carl, self
        original = carl._update_context

        def _update_context_then_seed():
            original()
            owner._seed_task_rng()

        carl._update_context = _update_context_then_seed
        # CWM's derivation: ONE task RNG per worker, seeded with the worker seed, whose j-th reset
        # draws episode j's initial state (protocol.worker_task_rng). CARL rebuilds the dm_control
        # task on context changes, so the same RNG object is handed to every rebuilt task.
        self._task_rng = P.worker_task_rng(self._drift_worker_seed)
        self._task_rng_armed = False

    def _seed_task_rng(self):
        if not self._task_rng_armed:
            return
        task = self._carl.env.env._task
        task._random = self._task_rng

    # -- gym API --------------------------------------------------------------------------------
    def reset(self, **kwargs):
        failures = self._physics_failures
        self._task_rng_armed = True
        self._seed_task_rng()                      # task kept (context unchanged) case
        obs, info = super().reset(**kwargs)        # rebuilt-task case seeds inside _update_context
        self._trace_reward = self._trace_factors = None
        if self._physics_failures != failures:
            raise RuntimeError("[drift_eval] PhysicsError during reset re-drew the episode; "
                               "an instrumented condition refuses unpaired replacements")
        if self._drift_mode == "scheduled":
            self._verify_applied(0)
        self.reset_canonical = self._canonical_row(0)
        if not self._external_recorder:
            self._recorder.start()
        return obs, info

    def step(self, action):
        # CWM timing: factor written, then physics.step with NO mj_forward (dmc_latent_factors.py
        # :440/:451). The CARL wrapper's `if switched_any: self._physics.forward()` is suppressed.
        physics = self._physics
        prev = physics.__dict__.get("forward")
        physics.forward = _no_forward
        try:
            obs, reward, terminated, truncated, info = super().step(action)
        finally:
            if prev is None:
                del physics.forward
            else:
                physics.forward = prev
        t = self._decision_step
        if self._drift_mode == "scheduled":
            self._verify_applied(t)
        self._last_canonical = self._canonical_row(t)
        if not self._external_recorder:
            self._recorder.append(reward, -1, self._last_canonical, t)
            if terminated or truncated:
                self._flush()
        return obs, reward, terminated, truncated, info

    def _flush(self):
        self._recorder.flush(self._canonical_factor_arrays()[1], self.trace_extras())

    def trace_extras(self):
        return {"pid": os.getpid(),
                "schedule_hash": self._drift_schedule_hash or "",
                "schedule_episode_index": (self.schedule_episode_index()
                                           if self._drift_mode == "scheduled" else -1),
                "worker_seed": self._drift_worker_seed,
                "schedule_clipped_entries": (self._sched_clipped_entries
                                             if self._drift_mode == "scheduled" else 0)}

    # -- values ---------------------------------------------------------------------------------
    def _canonical_row(self, t):
        """float64[4] in OUR units, canonical column order, NaN for inactive factors."""
        row = np.full(4, np.nan, dtype=np.float64)
        for i, name in enumerate(self._factor_names):
            fid = P.CARL_NAME_TO_FACTOR_ID[name]
            if self._drift_mode == "scheduled":
                row[fid] = self._sched_canon[self.schedule_episode_index()][name][t]
            else:
                row[fid] = P.carl_to_canonical(name, self._factor_values[i])
        return row

    def _verify_applied(self, t):
        ep = self._sched_canon[self.schedule_episode_index()]
        m = self._physics.model
        for i, name in enumerate(self._factor_names):
            want = P.canonical_to_carl(name, ep[name][t])
            ok = self._factor_values[i] == want
            if name == "gravity":
                ok = ok and m.opt.gravity[2] == -want
            elif name == "actuator_strength":
                ok = ok and np.array_equal(m.actuator_gear, self._default_actuator_gear * want)
            elif name == "joint_damping":
                ok = ok and np.array_equal(m.dof_damping, self._default_dof_damping * want)
            if not ok:
                raise RuntimeError(
                    f"[drift_eval] pid {os.getpid()} decision step {t}: {name} applied "
                    f"{self._factor_values[i]!r}, schedule says {want!r}")


class FixedObjectiveRewardSwitcher(CARLRewardModeSwitcher):
    """DALI's reward switcher. Under a schedule the objective is FIXED within each episode and taken
    from the generator's reward-label payload (befabde `drift_generator.build_reward_fixed_schedule`, the w2fmix
    rule: fixed within an episode, balanced across episodes). The label index uses the WORKER seed as
    ordinal offset, as CWM's reward wrapper does (dmc_reward_factors.py:376/:452, seeded with the
    worker seed at eval.py:111-118). Its RNG is seeded with the worker seed too (CWM eval.py:118),
    not DALI's seed+1000 (envs.py:954). Trace reward_mode_id = CWM's sorted-registry index."""

    def __init__(self, env, domain, mode_list, lambda_switch, lambda_switch_range, seed, *,
                 mode, physics_wrapper, reward_schedule=None, worker_seed=0, worker_index=0,
                 num_envs=1, trace_dir=None):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if mode == "scheduled" and reward_schedule is None:
            raise RuntimeError(
                f"[drift_eval] mixed bench in mode='scheduled' but no reward-label payload reached "
                f"this process (pid {os.getpid()}).")
        if mode == "bernoulli" and reward_schedule is not None:
            raise RuntimeError("[drift_eval] mode='bernoulli' was given a reward-label payload")
        super().__init__(env, domain=domain, mode_list=list(mode_list),
                         lambda_switch=lambda_switch, lambda_switch_range=lambda_switch_range,
                         seed=seed)
        if reward_schedule is not None:
            labels = list(reward_schedule["processes"]["reward"])
            if labels != list(mode_list):
                raise RuntimeError(f"[drift_eval] reward labels {labels} != wrapper modes "
                                   f"{list(mode_list)}")
        self._drift_mode = mode
        self._reward_schedule = reward_schedule
        self._phys = physics_wrapper
        self._phys._external_recorder = True
        self._drift_offset = int(worker_seed)
        self._drift_worker_index = int(worker_index)
        self._drift_num_envs = int(num_envs)
        self._recorder = _EpisodeRecorder(trace_dir, f"w{self._drift_worker_index}rf")
        self._registry_mode_ids = P.reward_mode_ids(domain)

    def reward_episode_index(self):
        return (self._drift_offset + self._episode_ordinal) % len(self._reward_schedule["episodes"])

    def fixed_mode(self):
        return self._reward_schedule["episodes"][self.reward_episode_index()]["reward"][0][1]

    def _get_scheduled_episode(self):
        # Overrides the base lookup so carl_reward_switch._ACTIVE_REWARD_SCHEDULE is never read.
        if self._drift_mode != "scheduled":
            return None
        return [[0, self.fixed_mode()]]

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        self._trace_reward = self._trace_mode_id = self._trace_factors = None
        if self._drift_mode == "scheduled" and self._current_mode != self.fixed_mode():
            raise RuntimeError("[drift_eval] fixed objective not installed at reset")
        self._recorder.start()
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self._drift_mode == "scheduled" and self._current_mode != self.fixed_mode():
            raise RuntimeError("[drift_eval] objective changed inside a scheduled episode")
        if self._decision_step != self._phys._decision_step:
            raise RuntimeError("[drift_eval] reward/physics decision steps diverged")
        self._recorder.append(reward, self._registry_mode_ids[self._current_mode],
                              self._phys._last_canonical, self._decision_step)
        if terminated or truncated:
            extras = {**self._phys.trace_extras(), "objective": self._current_mode,
                      "reward_schedule_hash": (self._reward_schedule or {}).get("schedule_hash", ""),
                      "reward_schedule_episode_index": (self.reward_episode_index()
                                                        if self._drift_mode == "scheduled" else -1)}
            self._recorder.flush(self._phys._canonical_factor_arrays()[1], extras)
        return obs, reward, terminated, truncated, info


# ── builders ─────────────────────────────────────────────────────────────────────────────────────
def _as_config(config):
    if isinstance(config, dict):
        from dreamerv3 import embodied
        return embodied.Config(config)
    return config


def benchmark_contexts(config, env_cls, task):
    """DALI envs.py:704-719 (`benchmark_discrete` branch of make_carl_env)."""
    from contextual_mbrl.dreamer import envs as E
    ctx = config.env.carl.context
    if "benchmark_discrete" not in ctx:
        raise ValueError(f"env.carl.context {ctx!r}: the §8b port supports benchmark_discrete only")
    parts = ctx.split("_", 2)
    split = parts[2] if len(parts) > 2 else "train"
    bm = E._BENCHMARK_CONTEXTS[task]
    default_ctx = env_cls.get_default_context()
    names = list(bm.keys())
    contexts = {}
    for combo in itertools.product(*[bm[f][split] for f in names]):
        c = dict(default_ctx)
        c.update(zip(names, combo))
        contexts[len(contexts)] = c
    return contexts


def check_bench_config(config, bench):
    b = P.BENCHES[bench]
    carl = config.env.carl
    got = (config.task, carl.regime, carl.regime_b_factors)
    want = (b["task"], "B", b["regime_b_factors"])
    if got != want:
        raise ValueError(f"[drift_eval] checkpoint config (task, regime, factors) = {got}, "
                         f"bench {bench} needs {want}")


def build_gym_env(config, bench, *, mode, schedule=None, reward_schedule=None, worker_index=0,
                  num_envs=1, base_seed, trace_dir=None, values=None):
    """Gymnasium-level chain (everything below FromGymnasium). Returns (env, physics_wrapper)."""
    from contextual_mbrl.dreamer import envs as E
    config = _as_config(config)
    check_bench_config(config, bench)
    b = P.BENCHES[bench]
    values = values if values is not None else P.load_values()
    _, task = config.task.split("_", 1)
    env_cls = E._TASK2ENV[task]
    contexts = benchmark_contexts(config, env_cls, task)
    seed = P.worker_seed(base_seed, worker_index)
    carl = config.env.carl

    # envs.py:825-870
    context_features = [o["context"] for o in E._TASK2CONTEXTS[task]]
    if carl.regime_b_factors == "K3" and "wind_x" not in context_features:
        context_features = context_features + ["wind_x"]
    env = env_cls(obs_context_as_dict=False, obs_context_features=context_features,
                  context_selector=E.RandomizedRoundRobinSelector(seed, contexts),
                  kwargs={"render_mode": "rgb_array"})
    env.reset(seed=seed)
    env.env.render_mode = "rgb_array"
    if task == "dmc_quadruped":
        _carl = env
        env.render = lambda: _carl.env.render(mode="rgb_array", camera_id=2)
    env = E.NormalizeContextWrapper(env)

    # envs.py:873-955
    split = getattr(carl, "regime_b_split", "train")
    lam = float(getattr(carl, "regime_b_lambda", 0.005))
    _lsr = getattr(carl, "regime_b_lambda_range", [0.003, 0.008])
    lsr = tuple(float(x) for x in _lsr) if _lsr else None
    rlam = float(getattr(carl, "regime_b_reward_lambda", lam))
    _rlsr = getattr(carl, "regime_b_reward_lambda_range", _lsr)
    rlsr = tuple(float(x) for x in _rlsr) if _rlsr else None
    fsplits = {}
    for entry in (getattr(carl, "regime_b_factor_splits", [""]) or []):
        if entry and "=" in entry:
            k, v = entry.split("=", 1)
            fsplits[k.strip()] = v.strip()
    native = _FACTORIES[(task, carl.regime_b_factors)](
        env, seed=seed, lambda_switch=lam, lambda_switch_range=lsr, split=split,
        factor_splits=fsplits or None)
    if sorted(native._factor_names) != sorted(b["carl_factors"]):
        raise ValueError(f"factory factors {native._factor_names} != bench {b['carl_factors']}")
    wind_key = b["values_keys"].get(3)
    phys = ScheduledCARLSwitching.from_native(
        native, seed, mode=mode, schedule=schedule, worker_index=worker_index,
        num_envs=num_envs, trace_dir=trace_dir,
        wind_table=values[wind_key] if wind_key else None)
    phys.use_cwm_value_tables(values, b["values_keys"], split, fsplits)
    env = phys
    if carl.regime_b_factors == "K3_mixed":
        modes = list(getattr(carl, "regime_b_reward_modes", ["walk_forward", "walk_backward"]))
        env = FixedObjectiveRewardSwitcher(
            env, domain="quadruped" if task == "dmc_quadruped" else "walker", mode_list=modes,
            lambda_switch=rlam, lambda_switch_range=rlsr, seed=seed, mode=mode,
            physics_wrapper=phys, reward_schedule=reward_schedule, worker_seed=seed,
            worker_index=worker_index, num_envs=num_envs, trace_dir=trace_dir)
    elif reward_schedule is not None:
        raise RuntimeError(f"[drift_eval] bench {bench} has no reward switcher; got a reward payload")
    return env, phys


def build_embodied_env(config, bench, **kw):
    """envs.py:960-963 on top of the gym chain (renders: needs a GL context)."""
    import dreamerv3
    from dreamerv3 import embodied
    from dreamerv3.embodied.envs import from_gymnasium
    from contextual_mbrl.dreamer import envs as E
    config = _as_config(config)
    env, _ = build_gym_env(config, bench, **kw)
    env = from_gymnasium.FromGymnasium(env, obs_key="obs")
    env = embodied.core.wrappers.RenderImage(env, key="image")
    env = E.ResizeImage(env)
    return dreamerv3.wrap_env(env, config)


def gym_env_fn(config_dict, bench, *, mode, schedule, worker_index, num_envs, base_seed,
               reward_schedule=None, trace_dir=None):
    """Picklable constructor: the payload travels INSIDE the partial into whichever process calls it."""
    return functools.partial(build_gym_env, config_dict, bench, mode=mode, schedule=schedule,
                             reward_schedule=reward_schedule, worker_index=worker_index,
                             num_envs=num_envs, base_seed=base_seed, trace_dir=trace_dir)


def embodied_env_fn(config_dict, bench, *, mode, schedule, worker_index, num_envs, base_seed,
                    reward_schedule=None, trace_dir=None):
    return functools.partial(build_embodied_env, config_dict, bench, mode=mode, schedule=schedule,
                             reward_schedule=reward_schedule, worker_index=worker_index,
                             num_envs=num_envs, base_seed=base_seed, trace_dir=trace_dir)
