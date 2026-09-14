"""Step-0 MuJoCo state equals CWM's for the same (worker, ordinal).

CWM: worker i's DMControl loads the dm_control task with task_kwargs={'random': base + i}
(envs/my_dmc.py:43-50, eval.py:108) and never reseeds it at reset (my_dmc.py:88-91), so episode j
starts from the j-th draw of RandomState(worker seed). Checked two ways: against `suite.load` with that
seed, and against the verifier's recorded CWM episodes (CausalWorldModel @befabde runner path, seed 1 -> 1007/1008).
"""
import os
import unittest

import numpy as np

from drift_eval.tests import _common as C
from drift_eval import envchain

VERIFIER_OURS = ("/home/jovyan/projects/CausalWorldModel/docs/trajd/drift_eval_port_2026-09-14/"
                 "verifier/out/ours")
BASE = 1007                 # trained seed 1 * 1000 + 7 (CWM eval.py:48)
WORKERS, EPISODES = (0, 1), 2


def port_initial_qpos(bench, worker):
    env, phys = envchain.build_gym_env(C.bench_config(bench, seed=1), bench, mode="scheduled",
                                       schedule=C.load_payload(bench, "ir_s"), worker_index=worker,
                                       num_envs=2, base_seed=BASE)
    out = []
    for _ in range(EPISODES):
        env.reset()
        out.append(phys._physics.data.qpos.copy())
        for _ in range(3):
            env.step(C.zero_action(env))
    return out


def cwm_initial_qpos(domain, worker):
    from dm_control import suite
    env = suite.load(domain_name=domain, task_name="walk", task_kwargs={"random": BASE + worker})
    out = []
    for _ in range(EPISODES):
        env.reset()
        out.append(env.physics.data.qpos.copy())
    return out


class ResetStateMatchesCWM(unittest.TestCase):

    def _check(self, bench, domain):
        recorded = None
        path = os.path.join(VERIFIER_OURS, f"{bench}__C__zero__native.npz")
        if os.path.exists(path):
            recorded = np.load(path)
        for w in WORKERS:
            port = port_initial_qpos(bench, w)
            suite_q = cwm_initial_qpos(domain, w)
            for j in range(EPISODES):
                self.assertTrue(np.array_equal(port[j], suite_q[j]), f"{bench} worker {w} ep {j}: suite")
                if recorded is not None:
                    k = [i for i in range(len(WORKERS) * EPISODES)
                         if int(recorded[f"e{i}_meta_worker"]) == w
                         and int(recorded[f"e{i}_meta_ordinal"]) == j]
                    self.assertEqual(len(k), 1)
                    self.assertTrue(np.array_equal(port[j], recorded[f"e{k[0]}_qpos"][0]),
                                    f"{bench} worker {w} ep {j}: verifier record")

    def test_walker_w3f(self):
        self._check("w3f", "walker")

    def test_quadruped_2f_phys(self):
        self._check("quadruped_2f_phys", "quadruped")


if __name__ == "__main__":
    unittest.main()
