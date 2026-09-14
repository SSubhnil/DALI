"""The realised episode reproduces the generator's own statistics.

* IR-A jump count read off the WRITTEN per-step trace (the file readers consume) equals the
  generator's `stats[physics_fK].jump_count`, and each trace column equals the schedule.
* Total variation of the realised IR-S and IR-A paths (natural coordinate, float64 wrapper state,
  reset value included) agree with each other and with the generator to 1e-9.
"""
import os
import tempfile
import unittest

import numpy as np

from drift_eval.tests import _common as C
from drift_eval import envchain, protocol as P


def _realise(bench, gen_cond, trace_dir):
    payload = C.load_payload(bench, gen_cond)
    env, phys = envchain.build_gym_env(C.bench_config(bench), bench, mode="scheduled",
                                       schedule=payload, reward_schedule=C.load_reward_payload(bench),
                                       worker_index=0, num_envs=1, base_seed=C.BASE_SEED,
                                       trace_dir=trace_dir)
    env.reset()
    snaps = [phys._factor_values.copy()]
    done = False
    while not done:
        _, _, te, tr, _ = env.step(C.zero_action(env))
        snaps.append(phys._factor_values.copy())
        done = te or tr
    names = list(phys._factor_names)
    paths = {}
    for i, name in enumerate(names):
        fid = P.CARL_NAME_TO_FACTOR_ID[name]
        paths[fid] = np.array([P.u_coordinate(fid, P.carl_to_canonical(name, s[i])) for s in snaps])
    ep_index = (C.BASE_SEED + 0) % payload["n_episodes"]
    return payload, paths, ep_index


class _Base:
    BENCH = None

    @classmethod
    def setUpClass(cls):
        from envs.schedule_channel import collect_traces
        cls.tmp = tempfile.mkdtemp(prefix=f"drift_eval_{cls.BENCH}_")
        cls.real = {}
        for cond in ("ir_s", "ir_a"):
            d = os.path.join(cls.tmp, cond)
            payload, paths, ep_index = _realise(cls.BENCH, cond, d)
            cls.real[cond] = dict(payload=payload, paths=paths, ep_index=ep_index,
                                  traces=collect_traces(d))

    def test_ir_a_jump_count_matches_generator(self):
        r = self.real["ir_a"]
        self.assertEqual(len(r["traces"]), 1)
        tr = r["traces"][0]
        ep = r["payload"]["episodes"][r["ep_index"]]
        for fid in sorted(r["paths"]):
            seq = ep[f"physics_f{fid}"]
            self.assertEqual(seq[0][1], seq[1][1],
                             "a jump at reset->step 1 would be invisible to the trace rows")
            col = tr["factor_values"][:, fid]
            self.assertTrue(np.array_equal(col, np.asarray([v for _, v in seq[1:]], np.float32)),
                            f"trace column {fid} differs from the schedule")
            u = np.array([P.u_coordinate(fid, float(v)) for v in col])
            jumps = int((np.abs(np.diff(u)) > 1e-6).sum())
            self.assertEqual(jumps, r["payload"]["stats"][f"physics_f{fid}"]["jump_count"],
                             f"factor {fid}")

    def test_ir_s_and_ir_a_total_variation_match_to_1e_9(self):
        s, a = self.real["ir_s"], self.real["ir_a"]
        for fid in sorted(s["paths"]):
            tv_s = float(np.abs(np.diff(s["paths"][fid])).sum())
            tv_a = float(np.abs(np.diff(a["paths"][fid])).sum())
            self.assertLessEqual(abs(tv_s - tv_a), 1e-9, f"factor {fid}: {tv_s} vs {tv_a}")
            gen_tv = s["payload"]["stats"][f"physics_f{fid}"]["tv"]
            self.assertLessEqual(abs(tv_s - gen_tv), 1e-9, f"factor {fid}: realised {tv_s} "
                                                            f"vs generator {gen_tv}")


class WalkerW3F(_Base, unittest.TestCase):
    BENCH = "w3f"


class Quadruped2FPhys(_Base, unittest.TestCase):
    BENCH = "quadruped_2f_phys"


if __name__ == "__main__":
    unittest.main()
