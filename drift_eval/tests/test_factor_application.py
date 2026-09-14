"""Schedule file -> the physics parameter used by each transition equals the schedule value, EXACTLY.

Expectations are written independently of the port, as CWM's applier computes them
(envs/dmc_latent_factors.py:257-274): clamp in our units (np.clip to [0.1, 10], wind to
±max|wind table|), then gravity = value * (-9.81), multiplicative = nominal * value, wind = Newtons on
the torso x-force. Covers reset (entry 0) and all 500 decision steps.
"""
import unittest

import numpy as np

from drift_eval.tests import _common as C
from drift_eval import envchain, protocol as P


class ScheduledFactorApplicationIsExact(unittest.TestCase):

    def _run(self, bench, gen_cond, worker_index=1):
        payload = C.load_payload(bench, gen_cond)
        rpay = C.load_reward_payload(bench)
        env, phys = envchain.build_gym_env(C.bench_config(bench), bench, mode="scheduled",
                                           schedule=payload, reward_schedule=rpay,
                                           worker_index=worker_index, num_envs=2,
                                           base_seed=C.BASE_SEED)
        wseed = C.BASE_SEED + worker_index
        ep = payload["episodes"][wseed % payload["n_episodes"]]
        env.reset()
        self._assert_physics(bench, phys, ep, 0)
        label = None
        if rpay is not None:
            label = rpay["episodes"][wseed % rpay["n_episodes"]]["reward"][0][1]
            self.assertEqual(env._current_mode, label)
        done = False
        for t in range(1, P.EPISODE_T + 1):
            self.assertFalse(done, f"episode ended before decision step {t}")
            _, _, te, tr, _ = env.step(C.zero_action(env))
            done = te or tr
            self._assert_physics(bench, phys, ep, t)
            if label is not None:
                self.assertEqual(env._current_mode, label)
        self.assertTrue(done, "episode did not end at decision step 500")

    def _assert_physics(self, bench, phys, ep, t):
        m, d = phys._physics.model, phys._physics.data
        for name in P.BENCHES[bench]["carl_factors"]:
            fid = {"actuator_strength": 0, "joint_damping": 0, "gravity": 1, "wind_x": 3}[name]
            bound = 120.0 if name == "wind_x" else None   # walker wind table max |v| (all splits)
            raw = ep[f"physics_f{fid}"][t][1]
            v = (float(np.clip(raw, -bound, bound)) if bound else float(np.clip(raw, 0.1, 10.0)))
            i = phys._factor_names.index(name)
            want_state = v * 9.81 if name == "gravity" else v
            self.assertEqual(phys._factor_values[i], want_state, f"{name} state at t={t}")
            if name == "gravity":
                self.assertEqual(m.opt.gravity[2], v * (-9.81), f"gravity at t={t}")
            elif name == "actuator_strength":
                self.assertTrue(np.array_equal(m.actuator_gear, phys._default_actuator_gear * v),
                                f"actuator gear at t={t}")
            elif name == "joint_damping":
                self.assertTrue(np.array_equal(m.dof_damping, phys._default_dof_damping * v),
                                f"dof damping at t={t}")
            elif name == "wind_x":
                self.assertEqual(d.xfrc_applied[phys._torso_body_id, 0], v, f"wind at t={t}")

    def test_walker_w3f_ir_a(self):
        self._run("w3f", "ir_a")

    def test_walker_w3f_h_s_beyond_dali_native_gravity_clip(self):
        self._run("w3f", "h_s")

    def test_walker_w2fmix_h_a(self):
        self._run("w2fmix", "h_a")

    def test_quadruped_2f_phys_h_s(self):
        self._run("quadruped_2f_phys", "h_s")

    def test_quadruped_2f_mixed_ir_s(self):
        self._run("quadruped_2f_mixed", "ir_s")


class ScheduleIsClampedExactlyLikeCWM(unittest.TestCase):
    """Out-of-range scheduled values are clamped as CWM's applier clamps them, so the applied vector
    matches CWM's bit for bit."""

    def test_quadruped_h_s_joint_damping_ulp_peak_becomes_10(self):
        payload = C.load_payload("quadruped_2f_phys", "h_s")
        ep_i = C.BASE_SEED % payload["n_episodes"]
        seq = payload["episodes"][ep_i]["physics_f0"]
        peaks = [t for t, v in seq if v > 10.0]
        self.assertTrue(peaks, "precondition: generator befabde H-S joint_damping exceeds 10.0")
        self.assertEqual(seq[peaks[0]][1], 10.000000000000002)
        env, phys = envchain.build_gym_env(C.bench_config("quadruped_2f_phys"), "quadruped_2f_phys",
                                           mode="scheduled", schedule=payload, worker_index=0,
                                           num_envs=1, base_seed=C.BASE_SEED)
        env.reset()
        i = phys._factor_names.index("joint_damping")
        for t in range(1, peaks[-1] + 1):
            env.step(C.zero_action(env))
            if t in peaks:
                self.assertEqual(phys._factor_values[i], 10.0, f"t={t}")
                self.assertEqual(phys._last_canonical[0], 10.0, f"t={t}")
                self.assertTrue(np.array_equal(phys._physics.model.dof_damping,
                                               phys._default_dof_damping * 10.0), f"t={t}")

    def _clamped(self, bench, key):
        ep = {k: [[t, 1.0] for t in range(P.EPISODE_T + 1)]
              for k in ("physics_f0", "physics_f1")}
        ep[key][1][1] = 10.5
        ep[key][2][1] = 0.05
        payload = {"T": P.EPISODE_T, "n_episodes": 1, "episodes": [ep], "schedule_hash": "clamp"}
        env, phys = envchain.build_gym_env(C.bench_config(bench), bench, mode="scheduled",
                                           schedule=payload, worker_index=0, num_envs=1,
                                           base_seed=C.BASE_SEED)
        env.reset()
        i = phys._factor_names.index("actuator_strength" if bench.startswith("w") else "joint_damping")
        env.step(C.zero_action(env))
        self.assertEqual(phys._factor_values[i], 10.0)
        env.step(C.zero_action(env))
        self.assertEqual(phys._factor_values[i], 0.1)

    def test_walker_actuator_clamped_to_cwm_bounds(self):
        self._clamped("w2fphys", "physics_f0")

    def test_quadruped_joint_damping_clamped_to_cwm_bounds(self):
        self._clamped("quadruped_2f_phys", "physics_f0")


class VendoredGeneratorShaIsEnforced(unittest.TestCase):

    def test_vendored_file_matches_vendored_txt(self):
        _, path, sha = P.load_generator()
        self.assertEqual(path, P.VENDORED_GENERATOR)
        self.assertEqual(sha, P.vendored_sha256())
        self.assertEqual(sha, "39b72cea67126f781605a2ebf7c658fa316ed380e9e2ee3badec132c561e09f6")

    def test_modified_copy_is_refused(self):
        import os
        import tempfile
        d = tempfile.mkdtemp(prefix="drift_eval_gen_")
        bad = os.path.join(d, "drift_generator.py")
        with open(P.VENDORED_GENERATOR) as f:
            src = f.read()
        with open(bad, "w") as f:
            f.write(src + "\n# one extra line\n")
        with self.assertRaisesRegex(RuntimeError, "refusing to load"):
            P.load_generator(bad)


if __name__ == "__main__":
    unittest.main()
