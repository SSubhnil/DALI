"""Outputs use CWM's schema: raw JSON (validated against a real CWM raw JSON when present), trace
cell arrays + sidecar, CSV columns (parsed from the pinned CWM runner). Also: the fail-closed trace
verification refuses a trace whose applied path differs from the schedule."""
import ast
import json
import os
import tempfile
import unittest

import numpy as np

from drift_eval.tests import _common as C
from drift_eval import outputs, protocol as P

REAL_RAW_JSON = ("/home/jovyan/projects/CausalWorldModel/docs/trajd/a2c_reads_2026-09-11/"
                 "ood_w2fmix_c2wm_final_2026-09-14/raw_ours/"
                 "trajd_c2wm_final_walker_mixed_s3__lambda_0.1.json")
PINNED_RUNNER = os.path.join(P.PINNED_CWM_ROOT, "benchmark", "run_eval_benchmark.py")
N, NUM_ENVS = 4, 2


def _synthetic_traces(payload, names):
    traces = []
    rng = np.random.RandomState(0)
    for w in range(NUM_ENVS):
        for j in range(N // NUM_ENVS):
            idx = (C.BASE_SEED + w + j) % payload["n_episodes"]
            fv = np.full((P.EPISODE_T, 4), np.nan, dtype=np.float32)
            for name in names:
                fid = P.CARL_NAME_TO_FACTOR_ID[name]
                fv[:, fid] = [v for _, v in payload["episodes"][idx][f"physics_f{fid}"][1:]]
            traces.append({"reward": rng.rand(P.EPISODE_T).astype(np.float32),
                           "reward_mode_id": np.full(P.EPISODE_T, -1, np.int16),
                           "factor_values": fv,
                           "factor_active": np.array([True, True, False, True]),
                           "decision_step": np.arange(1, P.EPISODE_T + 1, dtype=np.int32),
                           "schedule_hash": np.asarray(payload["schedule_hash"]),
                           "schedule_episode_index": np.asarray(idx),
                           "worker_tag": f"w{w}lf", "episode_ordinal": j})
    return traces


class OutputSchema(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="drift_eval_out_")
        self.payload = C.load_payload("w3f", "ir_a")
        self.names = P.BENCHES["w3f"]["carl_factors"]
        self.gen_sha = C.generator()[2]
        self.wind = P.load_values()["wind_walker"]

    def test_raw_json_matches_cwm_runner_schema(self):
        returns = [100.0, 200.5, 300.25, 400.0]
        rel = outputs.write_raw_json(os.path.join(self.tmp, "raw"), "dali_w3f_s3",
                                     "drift_ir_a", returns, None, seed=3, env="dm_walker_walk",
                                     git_sha="abc1234",
                                     extra_metadata={"action_mode": "sample",
                                                     "generator_sha256": self.gen_sha})
        with open(os.path.join(self.tmp, rel)) as f:
            got = json.load(f)
        self.assertEqual(tuple(got), outputs.RAW_JSON_KEYS)
        if os.path.exists(REAL_RAW_JSON):
            # A real CWM raw JSON written before befabde, plus the two fields it adds
            # (run_eval_benchmark.py:126 episode_keys, :132 metadata.paired_streams).
            with open(REAL_RAW_JSON) as f:
                ref = json.load(f)
            self.assertEqual(set(got), set(ref) | {"episode_keys"})
            self.assertTrue(set(ref["metadata"]) | {"paired_streams"} <= set(got["metadata"]))
            for k in ("row_id", "condition_id", "returns", "n_episodes"):
                self.assertIs(type(got[k]), type(ref[k]), k)
        self.assertEqual(got["returns"], returns)
        self.assertEqual(got["n_episodes"], len(returns))
        self.assertIsNone(got["reward_mse_per_episode"])
        self.assertEqual(got["metadata"]["action_mode"], "sample")
        self.assertEqual(got["metadata"]["generator_sha256"], self.gen_sha)

    def test_trace_cell_matches_cwm_cell_schema(self):
        from benchmark.switch_recovery import load_trace_cell
        traces = _synthetic_traces(self.payload, self.names)
        sums = [0.0] * N
        for tr in traces:
            k = int(tr["worker_tag"][1]) + tr["episode_ordinal"] * NUM_ENVS
            sums[k] = float(np.asarray(tr["reward"], np.float64).sum())
        keys = [(k % NUM_ENVS, k // NUM_ENVS) for k in range(N)]
        ordered, unjoined = outputs.order_and_verify(traces, return_keys=keys, suffix="lf",
                                                     payload=self.payload, carl_factors=self.names,
                                                     driver_returns=sums, wind_table=self.wind)
        self.assertEqual(unjoined, 0)
        meta = {k: "x" for k in outputs.CELL_META_KEYS}
        meta.update(action_mode="sample", generator_sha256=self.gen_sha)
        path = outputs.write_cell(os.path.join(self.tmp, "t__trace.npz"),
                                  outputs.build_cell(ordered, keys, sums), meta)
        cell = load_trace_cell(path)
        self.assertEqual(set(cell) - {"_meta"}, set(outputs.CELL_KEYS))
        self.assertEqual(cell["reward"].shape, (N, P.EPISODE_T))
        self.assertEqual(cell["reward"].dtype, np.float32)
        self.assertEqual(cell["reward_mode_id"].dtype, np.int16)
        self.assertEqual(cell["factor_values"].shape, (N, P.EPISODE_T, 4))
        self.assertEqual(cell["factor_values"].dtype, np.float32)
        self.assertEqual(cell["factor_active"].shape, (4,))
        self.assertEqual(cell["lambda_sampled"].dtype, np.float32)
        self.assertEqual(cell["schedule_id"].dtype, np.int32)
        for k, dt in (("episode_worker", np.int32), ("episode_ordinal", np.int32),
                      ("episode_return", np.float32)):
            self.assertEqual((cell[k].shape, cell[k].dtype), ((N,), dt), k)
        self.assertTrue(set(outputs.CELL_META_KEYS) <= set(cell["_meta"]))
        self.assertEqual(cell["_meta"]["generator_sha256"], self.gen_sha)

    def test_every_output_records_cwm_provenance_matching_the_root(self):
        import hashlib
        import subprocess
        from benchmark.switch_recovery import load_trace_cell
        gen_path = C.generator()[1]
        meta = outputs.provenance_metadata(gen_path, self.gen_sha)
        rel = outputs.write_raw_json(os.path.join(self.tmp, "raw"), "dali_w3f_s3", "phys_train",
                                     [1.0], None, seed=3, env="dm_walker_walk", git_sha="x",
                                     extra_metadata=meta)
        with open(os.path.join(self.tmp, rel)) as f:
            raw_meta = json.load(f)["metadata"]
        traces = _synthetic_traces(self.payload, self.names)
        keys = [(k % NUM_ENVS, k // NUM_ENVS) for k in range(N)]
        ordered, _ = outputs.order_and_verify(traces, return_keys=keys, suffix="lf",
                                              payload=self.payload, carl_factors=self.names,
                                              wind_table=self.wind)
        cell_path = outputs.write_cell(os.path.join(self.tmp, "p__trace.npz"),
                                       outputs.build_cell(ordered, keys, [0.0] * N), dict(meta))
        cell_meta = load_trace_cell(cell_path)["_meta"]
        notes = dict(kv.split("=", 1) for kv in outputs.provenance_notes(meta).split(";"))

        root = os.environ["CAUSAL_WORLD_MODEL_ROOT"]
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"]).decode().strip()
        want = {"cwm_root": root, "cwm_commit": commit}
        for field, relpath in {"cwm_sha256_experiments_dmc_yaml": "config_files/experiments/dmc.yaml",
                               "cwm_sha256_configure_yaml": "config_files/configure.yaml",
                               "cwm_sha256_latent_factor_values_yaml":
                                   "config_files/latent_factor_values.yaml"}.items():
            with open(os.path.join(root, relpath), "rb") as f:
                want[field] = hashlib.sha256(f.read()).hexdigest()
        for name, rec in (("raw JSON", raw_meta), ("cell sidecar", cell_meta), ("CSV notes", notes)):
            for field, value in want.items():
                self.assertIn(field, rec, f"{name} lacks {field}")
                self.assertEqual(str(rec[field]), str(value), f"{name} {field}")
            self.assertIn("cwm_dirty_count", rec, f"{name} lacks cwm_dirty_count")

    @unittest.skipUnless(os.path.exists(PINNED_RUNNER), "pinned CWM runner not on disk")
    def test_csv_columns_match_cwm_runner(self):
        with open(PINNED_RUNNER) as f:
            tree = ast.parse(f.read())
        cols = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CSV_COLUMNS"
                                                    for t in node.targets):
                cols = ast.literal_eval(node.value)
        self.assertEqual(outputs.CSV_COLUMNS, cols)

    def test_verification_refuses_a_trace_that_differs_from_the_schedule(self):
        traces = _synthetic_traces(self.payload, self.names)
        traces[1]["factor_values"][300, 1] *= np.float32(1.0001)
        with self.assertRaisesRegex(RuntimeError, "differs from the schedule"):
            outputs.order_and_verify(traces, return_keys=[(k % NUM_ENVS, k // NUM_ENVS)
                                                          for k in range(N)],
                                     suffix="lf", payload=self.payload, carl_factors=self.names,
                                     wind_table=self.wind)


if __name__ == "__main__":
    unittest.main()
