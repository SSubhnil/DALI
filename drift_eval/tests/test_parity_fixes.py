"""Env parity with CWM envs/dmc_latent_factors.py, one test per verifier finding (PARITY_TABLE.md).

1. timing  — no mj_forward between the factor write and physics.step (:440/:451);
2. gravity — Bernoulli value tables are CWM's multipliers × 9.81 computed as CWM computes them;
3. draw order — Bernoulli draws in factor-id order (:367/:407), replicated with CWM's RNG calls;
4. reward RNG — seeded with the worker seed (CWM eval.py:118), not seed + 1000.
"""
import unittest

import numpy as np

from drift_eval.tests import _common as C
from drift_eval import envchain, protocol as P


class FactorTimingHasNoForwardDuringStep(unittest.TestCase):

    def _check(self, bench):
        env, phys = envchain.build_gym_env(C.bench_config(bench), bench, mode="scheduled",
                                           schedule=C.load_payload(bench, "ir_s"),
                                           reward_schedule=C.load_reward_payload(bench),
                                           worker_index=0, num_envs=1, base_seed=C.BASE_SEED)
        env.reset()
        physics = phys._physics
        calls = [0]
        original = physics.forward

        def counting_forward(*a, **k):
            calls[0] += 1
            return original(*a, **k)

        physics.forward = counting_forward
        try:
            for _ in range(5):          # IR-S changes every factor at every decision step
                env.step(C.zero_action(env))
        finally:
            del physics.forward
        self.assertEqual(calls[0], 0, "physics.forward ran between a factor write and physics.step")

    def test_walker_w3f(self):
        self._check("w3f")

    def test_quadruped_2f_phys(self):
        self._check("quadruped_2f_phys")


class GravityValuesComputedAsCWM(unittest.TestCase):

    def _check(self, bench):
        env, phys = envchain.build_gym_env(C.bench_config(bench), bench, mode="bernoulli",
                                           worker_index=0, num_envs=1, base_seed=C.BASE_SEED)
        ours = [v * 9.81 for v in P.load_values()["gravity"]["train_values"]]
        self.assertNotEqual(ours, [3.924, 5.886, 9.810, 14.715, 19.620],
                            "precondition: CWM-computed and CARL-stored gravity differ in float")
        self.assertEqual(list(phys._factor_configs["gravity"].values), ours)
        env.reset()
        g = phys._physics.model.opt.gravity[2]
        self.assertIn(g, [v * (-9.81) for v in P.load_values()["gravity"]["train_values"]])

    def test_walker_w3f(self):
        self._check("w3f")

    def test_quadruped_2f_mixed(self):
        self._check("quadruped_2f_mixed")


class QuadrupedBernoulliDrawOrderMatchesCWM(unittest.TestCase):
    STEPS = 400

    def test_quadruped_2f_phys(self):
        values = P.load_values()
        jd = list(values["joint_damping"]["train_values"])
        g = list(values["gravity"]["train_values"])
        worker_index = 1
        env, phys = envchain.build_gym_env(C.bench_config("quadruped_2f_phys"), "quadruped_2f_phys",
                                           mode="bernoulli", worker_index=worker_index, num_envs=2,
                                           base_seed=C.BASE_SEED)
        self.assertEqual(phys._factor_names, ["joint_damping", "gravity"])
        # CWM DMControlLatentFactors with seed = worker seed: reset draws λ per active id, then an
        # initial value per active id (:356-375); each step draws random() then choice() per id (:407-411).
        rng = np.random.RandomState(C.BASE_SEED + worker_index)
        lam = [rng.uniform(0.003, 0.008) for _ in (0, 1)]
        cur = [rng.choice(jd), rng.choice(g)]
        env.reset()
        self.assertEqual(list(phys._current_lambdas), lam)
        self.assertEqual(list(phys._factor_values), [cur[0], cur[1] * 9.81])
        tables = [jd, g]
        for t in range(1, self.STEPS + 1):
            for i in (0, 1):
                if rng.random() < lam[i]:
                    cur[i] = rng.choice(tables[i])
            env.step(C.zero_action(env))
            self.assertEqual(list(phys._factor_values), [cur[0], cur[1] * 9.81], f"t={t}")


class RewardRngIsSeededWithTheWorkerSeed(unittest.TestCase):

    def test_quadruped_2f_mixed(self):
        worker_index = 1
        env, _ = envchain.build_gym_env(C.bench_config("quadruped_2f_mixed"), "quadruped_2f_mixed",
                                        mode="bernoulli", worker_index=worker_index, num_envs=2,
                                        base_seed=C.BASE_SEED)
        want = np.random.RandomState(C.BASE_SEED + worker_index).get_state()
        got = env._rng.get_state()
        self.assertTrue(np.array_equal(got[1], want[1]) and got[2:] == want[2:])


if __name__ == "__main__":
    unittest.main()
