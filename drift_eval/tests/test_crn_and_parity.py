"""CRN pairing and chain parity.

* Initial states: two conditions (training-protocol Bernoulli, IR-S, IR-A) at the same worker seed
  start every episode from the same MuJoCo state; a different worker starts elsewhere.
* Chain parity: the port's gym-level chain in Bernoulli mode carries the same wrapper configuration
  and RNG states as DALI's native `make_carl_env` (captured just before FromGymnasium, so no GL).
"""
import dataclasses
import unittest

import numpy as np

from drift_eval.tests import _common as C
from drift_eval import envchain


def _initial_qpos(bench, mode, payload, rpay, worker_index, resets=2):
    env, phys = envchain.build_gym_env(C.bench_config(bench), bench, mode=mode, schedule=payload,
                                       reward_schedule=rpay, worker_index=worker_index,
                                       num_envs=2, base_seed=C.BASE_SEED)
    out = []
    for _ in range(resets):
        env.reset()
        out.append(phys._physics.data.qpos.copy())
    return out


class InitialStatesArePairedAcrossConditions(unittest.TestCase):

    def _check(self, bench):
        rpay = C.load_reward_payload(bench)
        a = _initial_qpos(bench, "bernoulli", None, None, 1)
        s = _initial_qpos(bench, "scheduled", C.load_payload(bench, "ir_s"), rpay, 1)
        b = _initial_qpos(bench, "scheduled", C.load_payload(bench, "ir_a"), rpay, 1)
        other = _initial_qpos(bench, "bernoulli", None, None, 0)
        for k in range(2):
            self.assertTrue(np.array_equal(a[k], s[k]), f"episode {k}: A vs IR-S")
            self.assertTrue(np.array_equal(s[k], b[k]), f"episode {k}: IR-S vs IR-A")
            self.assertFalse(np.array_equal(a[k], other[k]), f"episode {k}: workers 0/1 identical")

    def test_walker_w3f(self):
        self._check("w3f")

    def test_quadruped_2f_phys(self):
        self._check("quadruped_2f_phys")


class _Captured(Exception):
    pass


def _native_gym_env(config):
    from contextual_mbrl.dreamer import envs as E
    original = E.from_gymnasium.FromGymnasium

    def capture(env, obs_key="obs"):
        raise _Captured(env)

    E.from_gymnasium.FromGymnasium = capture
    try:
        E.make_carl_env(config)
    except _Captured as c:
        return c.args[0]
    finally:
        E.from_gymnasium.FromGymnasium = original
    raise AssertionError("native chain never reached FromGymnasium")


def _chain(env):
    out = []
    while env is not None:
        out.append(env)
        env = getattr(env, "env", None) if not hasattr(env, "context_selector") else None
    return out


def _rng_equal(r1, r2):
    s1, s2 = r1.get_state(), r2.get_state()
    return s1[0] == s2[0] and np.array_equal(s1[1], s2[1]) and s1[2:] == s2[2:]


class ChainParityWithDaliNative(unittest.TestCase):
    """Parity with DALI's native chain EXCEPT the three deliberate CWM-parity departures, which are
    asserted here in their CWM form: factor order by factor id, gravity table computed as CWM
    (value × 9.81), reward RNG seeded with the worker seed (see test_parity_fixes)."""

    def _check(self, bench):
        from dreamerv3 import embodied
        from drift_eval import protocol as P
        config = embodied.Config(C.bench_config(bench, seed=5))
        native = _chain(_native_gym_env(config))
        port_env, _ = envchain.build_gym_env(config, bench, mode="bernoulli", worker_index=0,
                                             num_envs=1, base_seed=5)
        port = _chain(port_env)
        self.assertEqual(len(native), len(port))
        for n, p in zip(native, port):
            self.assertIsInstance(p, type(n))
            if hasattr(n, "_factor_configs"):
                self.assertEqual(p._factor_names, sorted(n._factor_names,
                                                         key=lambda x: P.CARL_NAME_TO_FACTOR_ID[x]))
                for name in n._factor_names:
                    nc, pc = n._factor_configs[name], p._factor_configs[name]
                    if name == "gravity":
                        self.assertEqual(list(pc.values), [v * 9.81 for v in
                                                           P.load_values()["gravity"]["train_values"]])
                        pc = dataclasses.replace(pc, values=nc.values)
                    self.assertEqual(nc, pc, name)
                    i, j = n._factor_names.index(name), p._factor_names.index(name)
                    self.assertEqual((n._train_low[i], n._train_high[i], n._current_lambdas[i]),
                                     (p._train_low[j], p._train_high[j], p._current_lambdas[j]))
                self.assertEqual((n._action_repeat, n._initial_mode),
                                 (p._action_repeat, p._initial_mode))
                self.assertTrue(_rng_equal(n._rng, p._rng), "physics wrapper RNG")
            if hasattr(n, "_mode_list"):
                self.assertEqual((n._mode_list, n._lambda_switch, n._lambda_switch_range),
                                 (p._mode_list, p._lambda_switch, p._lambda_switch_range))
            if hasattr(n, "context_selector"):
                self.assertEqual(list(n.obs_context_features), list(p.obs_context_features))
                self.assertEqual(n.context_selector.contexts, p.context_selector.contexts)
                self.assertEqual(n.context_selector.rand.getstate(),
                                 p.context_selector.rand.getstate())

    def test_all_benches(self):
        for bench in ("w3f", "w2fphys", "w2fmix", "quadruped_2f_phys", "quadruped_2f_mixed"):
            with self.subTest(bench=bench):
                self._check(bench)


if __name__ == "__main__":
    unittest.main()
