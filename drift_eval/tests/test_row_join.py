"""A REAL port output (gym-level episodes, the port's own writer path) joins trace rows to returns by
(worker, ordinal) exactly as CWM's runner does (`_join_traces_to_returns`, mirrored verbatim in
drift_eval/outputs.py from CausalWorldModel @befabde run_eval_benchmark.py:659-688). Every row is
checked against rewards recorded independently in the stepping loop; not a field-presence check."""
import json
import os
import tempfile
import unittest

import numpy as np

from drift_eval.tests import _common as C
from drift_eval import envchain, outputs, protocol as P

N_EPISODES, NUM_ENVS, ROUNDS = 3, 2, 2       # 4 episodes run, 3 retained: one unjoined trace


class TraceRowsJoinReturnsByKey(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from envs.schedule_channel import collect_traces
        cls.tmp = tempfile.mkdtemp(prefix="drift_eval_join_")
        wdir = os.path.join(cls.tmp, "traces", "_workers", "cell")
        payload = C.load_payload("w3f", "ir_a")
        per_worker, cls.rewards, cls.return64 = {}, {}, {}
        for w in range(NUM_ENVS):
            env, _ = envchain.build_gym_env(C.bench_config("w3f"), "w3f", mode="scheduled",
                                            schedule=payload, worker_index=w, num_envs=NUM_ENVS,
                                            base_seed=C.BASE_SEED, trace_dir=wdir)
            rng = np.random.RandomState(100 + w)
            per_worker[w] = []
            for j in range(ROUNDS):
                env.reset()
                rows, done = [], False
                while not done:
                    a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
                    _, r, te, tr, _ = env.step(a)
                    rows.append(r)
                    done = te or tr
                cls.rewards[(w, j)] = np.asarray(rows, dtype=np.float32)
                cls.return64[(w, j)] = float(np.asarray(rows, dtype=np.float64).sum())
                per_worker[w].append(cls.return64[(w, j)])
        cls.returns = outputs.select_plan_episodes(per_worker, C.BASE_SEED, N_EPISODES, NUM_ENVS)
        cls.keys = outputs.plan_keys(C.BASE_SEED, N_EPISODES, NUM_ENVS)
        cls.worker_traces = collect_traces(wdir)
        values = P.load_values()
        cls.raw_path, cls.cell_path = outputs.write_condition_outputs(
            cls.tmp, bench="w3f", row_id="dali_w3f_s3", condition_id="drift_ir_a", axis="drift",
            seed=3, git_sha="x", returns=cls.returns, return_keys=cls.keys, mse=None,
            traces=cls.worker_traces, payload=payload, reward_payload=None, labels=None,
            carl_factors=P.BENCHES["w3f"]["carl_factors"], wind_table=values["wind_walker"],
            extra={}, notes="")

    def test_every_row_joins_to_its_return_by_worker_and_ordinal(self):
        from benchmark.switch_recovery import load_trace_cell
        with open(self.raw_path) as f:
            raw = json.load(f)
        cell = load_trace_cell(self.cell_path)
        try:
            joined, unjoined = outputs.join_traces_to_returns(
                [t for t in self.worker_traces if str(t["worker_tag"]).endswith("lf")],
                raw["episode_keys"], "lf", "drift_ir_a")
        except SystemExit as exc:
            self.fail(f"CWM's keyed join refused the port output: {exc}")
        self.assertEqual(unjoined, 1)
        self.assertEqual(cell["_meta"]["unjoined_worker_traces"], 1)
        self.assertEqual(len(raw["returns"]), N_EPISODES)
        for i, key in enumerate(raw["episode_keys"]):
            key = tuple(key)
            self.assertEqual((int(cell["episode_worker"][i]), int(cell["episode_ordinal"][i])), key,
                             f"row {i}")
            self.assertTrue(np.array_equal(cell["reward"][i], self.rewards[key]), f"row {i} {key}")
            self.assertTrue(np.array_equal(joined[i]["reward"], cell["reward"][i]), f"row {i}")
            self.assertEqual(raw["returns"][i], self.return64[key], f"row {i}")
            self.assertEqual(cell["episode_return"][i], np.float32(raw["returns"][i]), f"row {i}")


class PairedStreamsIsNullWithReason(TraceRowsJoinReturnsByKey):
    """DALI's eval has no first-action sampler for CWM's PairedStreams to seed, so every output of the
    real writer path carries paired_streams = null and its reason, never true/false."""

    def test_every_row_joins_to_its_return_by_worker_and_ordinal(self):
        pass        # inherited setUpClass only; the join itself is tested once, above

    def test_paired_streams_null_with_reason_in_raw_json_cell_meta_and_csv(self):
        import csv
        from benchmark.switch_recovery import load_trace_cell
        reason = "not_applicable: port has no first-action sampler"
        with open(self.raw_path) as f:
            meta = json.load(f)["metadata"]
        self.assertIn("paired_streams", meta)
        self.assertIsNone(meta["paired_streams"])
        self.assertEqual(meta["paired_streams_reason"], reason)
        cmeta = load_trace_cell(self.cell_path)["_meta"]
        self.assertIn("paired_streams", cmeta)
        self.assertIsNone(cmeta["paired_streams"])
        self.assertEqual(cmeta["paired_streams_reason"], reason)
        with open(os.path.join(self.tmp, "main_results.csv")) as f:
            notes = list(csv.DictReader(f))[-1]["notes"]
        self.assertIn("paired_streams=null", notes)
        self.assertIn(f"paired_streams_reason={reason}", notes)
        self.assertNotIn("paired_streams=False", notes)
        self.assertNotIn("paired_streams=True", notes)


if __name__ == "__main__":
    unittest.main()
