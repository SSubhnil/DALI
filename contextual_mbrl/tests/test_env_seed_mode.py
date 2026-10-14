"""envs.seed_mode: in-process envs of one batch get distinct seeds under `index` (eval), and
`proc` (default, training) keeps the legacy shared seed.

Defect (2026-10-14, /home/jovyan/dali_label_check_2026-10-14): seed = proc_id + config.seed with
proc_id 0 for every env under envs.parallel none, so the N envs of a batch replayed one factor
schedule; a 100-episode eval cell held only 25 distinct schedules x 4 copies.
"""
import unittest

from contextual_mbrl.tests import _envtools as T  # noqa: F401  (sets CPU / no-GL env first)
from contextual_mbrl.dreamer.envs import gen_carl_val_envs, make_envs

STEPS = 600          # > one 500-decision episode, so every env also resamples at a reset
BENCHES = ("walker_2f", "walker_3f", "walker_mixed", "quad_2f", "quad_mixed")


def _index_mode(cfg):
    # On a tree without envs.seed_mode the key is unknown; the batch then runs as before.
    try:
        return cfg.update({"envs.seed_mode": "index"})
    except KeyError:
        return cfg


def _schedules(cfg):
    batch = make_envs(cfg)
    try:
        return [T.rollout(e, STEPS)[:2] for e in batch._envs]
    finally:
        batch.close()


class InProcessEvalEnvsAreDistinct(unittest.TestCase):

    def test_index_mode_distinct_schedules_every_bench(self):
        for bench in BENCHES:
            with self.subTest(bench=bench):
                scheds = _schedules(_index_mode(T.bench_config(bench, amount=4)))
                keys = [repr(s) for s in scheds]
                self.assertEqual(len(set(keys)), 4,
                                 f"{bench}: {4 - len(set(keys)) + 1} of 4 in-process envs replay one "
                                 f"factor/reward schedule")

    def test_val_envs_distinct_even_in_proc_mode(self):
        # gen_carl_val_envs (DALI's own eval.py / eval_record.py) forces per-env seeds whatever
        # envs.seed_mode says. Regime-B config so each env's seed drives a factor schedule.
        cfg = T.bench_config("walker_2f", amount=3).update({"env.carl.context": "specific_0-1.0"})
        self.assertEqual(getattr(cfg.envs, "seed_mode", "proc"), "proc")
        batch, _ = next(gen_carl_val_envs(cfg))
        try:
            keys = [repr(T.rollout(e, STEPS)[:2]) for e in batch._envs]
        finally:
            batch.close()
        self.assertEqual(len(set(keys)), 3, "val envs replay one factor schedule")


class TrainingDefaultUnchanged(unittest.TestCase):

    def test_default_is_proc_and_shared_in_process(self):
        cfg = T.bench_config("walker_2f", amount=3)
        self.assertEqual(getattr(cfg.envs, "seed_mode", "proc"), "proc")
        scheds = _schedules(cfg)
        self.assertTrue(all(repr(s) == repr(scheds[0]) for s in scheds),
                        "default (training) seeding must reproduce the legacy shared seed")

    def test_env0_seed_identical_across_modes(self):
        from contextual_mbrl.dreamer.envs import _env_seed
        cfg = T.bench_config("walker_2f")
        self.assertEqual(_env_seed(cfg, 0, "index"), _env_seed(cfg, 0, "proc"))
        self.assertEqual([_env_seed(cfg, i, "index") for i in range(4)], [11, 12, 13, 14])
        with self.assertRaises(ValueError):
            _env_seed(cfg, None, "index")


if __name__ == "__main__":
    unittest.main()
