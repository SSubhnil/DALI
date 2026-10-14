"""env.carl.physics_error: in eval a MuJoCo divergence ends the episode and its return so far is
scored (TK/DRAMA semantics, CWM envs/my_dmc.py DMControl.step); training default unchanged.

Defect (2026-10-14, /home/jovyan/parity_audit_2026-10-14/eval_parity.csv row E1): a PhysicsError
reached RestartOnException, the env was rebuilt and the partial episode dropped, so an eval cell
replaced diverged episodes (biased up) where TK counts them. Divergence is forced by patching the
CARL env's control step (or the regime-B wrapper's step) to raise dm_control's PhysicsError.
"""
import unittest

import numpy as np
from dm_control.rl.control import PhysicsError

from contextual_mbrl.tests import _envtools as T
from contextual_mbrl.dreamer.envs import gen_carl_val_envs, make_envs

DECISION = 11        # the decision at which the env diverges (1-based, after the reset step)


def _terminate(cfg):
    try:
        return cfg.update({"env.carl.physics_error": "terminate"})
    except KeyError:     # tree without the key: legacy restart path
        return cfg


def _carl(env):
    return next(x for x in T.chain(env) if type(x).__name__.startswith("CARLDmc"))


def _inject_substep(env, decision):
    """Raise on the 2nd control sub-step of `decision`; record each real sub-step reward."""
    carl = _carl(env)
    real = carl.step
    log = {"calls": 0, "rewards": []}

    def step(action):
        log["calls"] += 1
        if log["calls"] == 2 * decision:
            raise PhysicsError("injected mjWARN_BADQACC")
        out = real(action)
        log["rewards"].append(float(out[1]))
        return out
    carl.step = step
    return log


def _inject_wrapper(env, decision):
    """Raise from the regime-B wrapper's step (its post-switch mj_forward site) at `decision`."""
    phys = T.find(env, "CARLIntraEpisodeSwitching")
    real = phys.step
    log = {"calls": 0}

    def step(action):
        log["calls"] += 1
        if log["calls"] == decision:
            raise PhysicsError("injected forward() divergence")
        return real(action)
    phys.step = step
    return log


def _run(env, inject, decision=DECISION):
    act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
    act["reset"] = True
    obs = env.step(act)                 # reset
    inner_before = env.env              # RestartOnException rebuilds env.env on a crash
    log = inject(env, decision)
    act["reset"] = False
    rng = np.random.RandomState(0)
    steps = []
    for _ in range(decision):
        act["action"] = rng.uniform(-1, 1, env.act_space["action"].shape).astype(np.float32)
        obs = env.step(act)
        steps.append({k: obs[k] for k in ("is_first", "is_last", "is_terminal", "reward")})
        if obs["is_last"] or obs["is_first"]:
            break
    return steps, log, env.env is inner_before


class EvalCountsDivergedEpisode(unittest.TestCase):

    def _assert_counted(self, steps, rebuilt_not, label, reward):
        last = steps[-1]
        self.assertEqual(len(steps), DECISION, f"{label}: episode not ended at the diverging decision")
        self.assertTrue(bool(last["is_last"]) and bool(last["is_terminal"]) and not bool(last["is_first"]),
                        f"{label}: diverged step is not a terminal of the same episode (replaced?)")
        self.assertTrue(rebuilt_not, f"{label}: env was rebuilt (episode replaced)")
        self.assertAlmostEqual(float(last["reward"]), reward, places=5)

    def test_substep_divergence_terminates_and_scores(self):
        for bench in ("walker_2f", "walker_mixed", "quad_2f", "quad_mixed"):
            with self.subTest(bench=bench):
                batch = make_envs(_terminate(T.bench_config(bench, amount=1, **{"envs.restart": True})))
                try:
                    steps, log, same = _run(batch._envs[0], _inject_substep)
                finally:
                    batch.close()
                # reward of the diverging decision = its first (completed) sub-step, as TK.
                self._assert_counted(steps, same, bench, log["rewards"][-1] if log["rewards"] else 0.0)

    def test_wrapper_level_divergence_terminates(self):
        for bench in ("walker_2f", "quad_mixed"):
            with self.subTest(bench=bench):
                batch = make_envs(_terminate(T.bench_config(bench, amount=1, **{"envs.restart": True})))
                try:
                    steps, _, same = _run(batch._envs[0], _inject_wrapper)
                finally:
                    batch.close()
                self._assert_counted(steps, same, bench, 0.0)

    def test_val_envs_terminate_by_default(self):
        cfg = T.bench_config("quad_2f", amount=1, **{"envs.restart": True,
                                                      "env.carl.context": "specific_0-1.0"})
        batch, _ = next(gen_carl_val_envs(cfg))
        try:
            steps, log, same = _run(batch._envs[0], _inject_substep)
        finally:
            batch.close()
        self._assert_counted(steps, same, "gen_carl_val_envs", log["rewards"][-1])


class TrainingDefaultUnchanged(unittest.TestCase):

    def test_default_restarts_and_replaces(self):
        cfg = T.bench_config("walker_2f", amount=1, **{"envs.restart": True})
        self.assertEqual(getattr(cfg.env.carl, "physics_error", "restart"), "restart")
        batch = make_envs(cfg)
        try:
            steps, _, same = _run(batch._envs[0], _inject_substep)
        finally:
            batch.close()
        self.assertFalse(same, "default must keep RestartOnException's rebuild")
        self.assertTrue(bool(steps[-1]["is_first"]), "default must restart the episode")


if __name__ == "__main__":
    unittest.main()
