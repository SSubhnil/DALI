"""Output fields match CWM's on every item the verifier listed: condition ids, reward-mode ids,
the mixed-cell schedule hash, experiment/env, std ddof 0, envs per batch and the CRN plan."""
import os
import unittest

import numpy as np

from drift_eval.tests import _common as C
from drift_eval import envchain, outputs, protocol as P

# CWM's condition ids at the commit the port cites, read from CAUSAL_WORLD_MODEL_ROOT (the root the port
# already requires). The root must CONTAIN the commit (befabde is an ancestor of its HEAD): a shared object
# store alone would let `git show` succeed from an older checkout.
CWM_COMMIT = "befabde38bfea2e9d150fa2da41facf738c49d59"
CWM_CONDITIONS_SHA256 = "805b26781c581fbc17ddf62b88ce0880f744c45511803ebca072b3f5afa84ccb"


def _cwm_conditions_yaml():
    """(text, None) or (None, skip reason)."""
    import subprocess
    root = P.cwm_root()
    try:
        subprocess.check_call(["git", "-C", root, "merge-base", "--is-ancestor", CWM_COMMIT, "HEAD"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        text = subprocess.check_output(["git", "-C", root, "show",
                                        f"{CWM_COMMIT}:benchmark/eval_conditions.yaml"],
                                       stderr=subprocess.DEVNULL).decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, (f"CAUSAL_WORLD_MODEL_ROOT={root} does not contain CausalWorldModel commit "
                      f"{CWM_COMMIT} (required for benchmark/eval_conditions.yaml)")
    return text, None


class SchemaMatchesCWM(unittest.TestCase):

    def test_condition_ids_are_cwm_ids(self):
        want = {"A": "phys_train", "B": "drift_ir_a", "C": "drift_ir_s", "D": "drift_h_a",
                "E": "drift_h_s", **{f"F{t}": f"drift_tau_{t}" for t in P.TAUS}}
        self.assertEqual({k: v["id"] for k, v in P.CONDITIONS.items()}, want)

    def test_condition_ids_exist_in_cwm_eval_conditions_at_the_cited_commit(self):
        import hashlib
        import yaml
        text, reason = _cwm_conditions_yaml()
        if text is None:
            self.skipTest(reason)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), CWM_CONDITIONS_SHA256)
        fams = yaml.safe_load(text)
        ours = {c["id"] for fam in ("dmc_physics_3f", "dmc_drift_walker_2f_mixed") for c in fams[fam]}
        want = {v["id"] for v in P.CONDITIONS.values()}
        self.assertTrue(want <= ours, want - ours)

    def test_reward_mode_ids_are_cwm_registry_indices(self):
        self.assertEqual(P.reward_mode_ids("walker")["walk_forward"], 6)
        self.assertEqual(P.reward_mode_ids("walker")["walk_backward"], 5)
        self.assertEqual(P.reward_mode_ids("quadruped")["go_north"], 3)
        self.assertEqual(P.reward_mode_ids("quadruped")["go_south"], 4)
        rpay = C.load_reward_payload("quadruped_2f_mixed")
        env, _ = envchain.build_gym_env(C.bench_config("quadruped_2f_mixed"), "quadruped_2f_mixed",
                                        mode="scheduled",
                                        schedule=C.load_payload("quadruped_2f_mixed", "ir_a"),
                                        reward_schedule=rpay, worker_index=0, num_envs=1,
                                        base_seed=C.BASE_SEED)
        env.reset()
        done = False
        while not done:
            _, _, te, tr, _ = env.step(C.zero_action(env))
            done = te or tr
        trace = env._recorder.in_memory[0]
        label = rpay["episodes"][C.BASE_SEED % C.N_EPISODES]["reward"][0][1]
        self.assertEqual(set(trace["reward_mode_id"].tolist()),
                         {P.reward_mode_ids("quadruped")[label]})

    def test_mixed_cell_schedule_hash_is_the_reward_schedule_hash(self):
        from envs.schedule_channel import crn_episode_plan
        phys = C.load_payload("w2fmix", "ir_s")
        rpay = C.load_reward_payload("w2fmix")
        labels = outputs.label_plan(rpay, C.BASE_SEED, 10, 5)
        meta = outputs.cell_meta(bench="w2fmix", condition_id="drift_ir_s", axis="drift",
                                 row_id="r", seed=3, git_sha="x", physics_payload=phys,
                                 reward_payload=rpay, labels=labels,
                                 extra={"schedule_hash": "must-not-win"})
        self.assertEqual(meta["schedule_hash"], rpay["schedule_hash"])
        self.assertEqual(meta["physics_schedule_hash"], phys["schedule_hash"])
        lab = [e["reward"][0][1] for e in rpay["episodes"]]
        want = [lab[(p["worker_seed"] + p["episode_in_worker"]) % len(lab)]
                for p in crn_episode_plan(C.BASE_SEED, 10, 5)]
        self.assertEqual(meta["reward_label_plan"], want)
        phys_only = outputs.cell_meta(bench="w3f", condition_id="drift_ir_s", axis="drift",
                                      row_id="r", seed=3, git_sha="x",
                                      physics_payload=C.load_payload("w3f", "ir_s"),
                                      reward_payload=None)
        self.assertEqual(phys_only["schedule_hash"], C.load_payload("w3f", "ir_s")["schedule_hash"])
        self.assertNotIn("physics_schedule_hash", phys_only)

    def test_row_experiment_and_env_are_cwm_names(self):
        row = outputs.csv_row(bench="w3f", condition_id="drift_ir_s", axis="drift", row_id="r",
                              seed=1, returns=[1.0, 2.0], mse=None, raw_rel="raw/x.json",
                              git_sha="x", notes="")
        self.assertEqual((row["experiment"], row["env"], row["domain"]),
                         ("walker_3f_phys", "dm_walker_walk", "dmc"))
        q = outputs.csv_row(bench="quadruped_2f_mixed", condition_id="phys_train", axis="physics",
                            row_id="r", seed=1, returns=[1.0], mse=None, raw_rel="", git_sha="x",
                            notes="")
        self.assertEqual((q["experiment"], q["env"]), ("quadruped_2f_mixed", "dm_quadruped_walk"))

    def test_std_return_is_ddof_0(self):
        returns = [10.0, 20.0, 40.0]
        row = outputs.csv_row(bench="w3f", condition_id="drift_ir_s", axis="drift", row_id="r",
                              seed=1, returns=returns, mse=None, raw_rel="", git_sha="x", notes="")
        self.assertEqual(row["std_return"], float(np.std(np.asarray(returns, np.float32).astype(np.float64))))
        self.assertAlmostEqual(row["std_return"], 12.472191289246473, places=12)

    def test_envs_per_batch_is_5_and_the_retained_plan_is_cwms(self):
        self.assertEqual(P.DEFAULT_NUM_ENVS, 5)
        n, e = 64, P.DEFAULT_NUM_ENVS
        per_worker = {w: [(w, j) for j in range(-(-n // e))] for w in range(e)}   # 13 rounds
        got = outputs.select_plan_episodes(per_worker, C.BASE_SEED, n, e)
        self.assertEqual(got, [(k % e, k // e) for k in range(n)])
        self.assertNotIn((4, 12), got, "the 65th completion must be dropped")


if __name__ == "__main__":
    unittest.main()
