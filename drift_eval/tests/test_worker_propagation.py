"""The schedule reaches the process that STEPS the env; missing payloads fail closed.

Children run at module level (spawn pickles targets by reference) and receive the env constructor
as a pickled partial, exactly as a spawned env worker would.
"""
import multiprocessing as mp
import os
import unittest

from drift_eval.tests import _common as C
from drift_eval import envchain


def _child_module_global(conn):
    from drift_eval import protocol as P
    P.cwm_root()
    from benchmark.wrappers import carl_intra_episode as cie
    conn.send({"child_schedule_is_none": cie._ACTIVE_FACTOR_SCHEDULE is None})
    conn.close()


def _child_build_and_step(conn, env_fn, steps):
    try:
        env, phys = env_fn()
        env.reset()
        for _ in range(steps):
            env.step(C.zero_action(env))
        conn.send({"ok": True, "pid": os.getpid(), "mode": phys._drift_mode,
                   "hash": phys._drift_schedule_hash, "t": phys._decision_step,
                   "row": phys._last_canonical.tolist(), "ep_index": phys.schedule_episode_index(),
                   "objective": getattr(env, "_current_mode", None)})
    except Exception as exc:  # the parent asserts on the message
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
    conn.close()


def _run_child(target, *args, timeout=300):
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(send, *args))
    proc.start()
    send.close()
    if not recv.poll(timeout):
        proc.terminate()
        raise AssertionError(f"child produced nothing within {timeout}s")
    out = recv.recv()
    proc.join(60)
    return out


class ParentGlobalDefectFixture(unittest.TestCase):
    def test_parent_module_global_does_not_cross_spawn(self):
        """The defect the port must not depend on: the DALI-side wrapper's global stays None."""
        from benchmark.wrappers import carl_intra_episode as cie
        cie.set_factor_schedule(C.load_payload("w3f", "ir_a"))
        try:
            self.assertTrue(_run_child(_child_module_global)["child_schedule_is_none"])
        finally:
            cie.clear_instrumentation()


class ScheduleReachesTheSteppingProcess(unittest.TestCase):
    STEPS = 5

    def _check(self, bench):
        payload = C.load_payload(bench, "ir_a")
        rpay = C.load_reward_payload(bench)
        fn = envchain.gym_env_fn(C.bench_config(bench), bench, mode="scheduled", schedule=payload,
                                 reward_schedule=rpay, worker_index=1, num_envs=2,
                                 base_seed=C.BASE_SEED)
        out = _run_child(_child_build_and_step, fn, self.STEPS)
        self.assertTrue(out["ok"], out.get("error"))
        self.assertNotEqual(out["pid"], os.getpid())
        self.assertEqual(out["mode"], "scheduled")
        self.assertEqual(out["hash"], payload["schedule_hash"])
        self.assertEqual(out["t"], self.STEPS)
        ep = payload["episodes"][out["ep_index"]]
        for key, seq in ep.items():
            fid = int(key[len("physics_f"):])
            self.assertEqual(out["row"][fid], seq[self.STEPS][1], key)
        if rpay is not None:
            self.assertEqual(out["objective"],
                             rpay["episodes"][(C.BASE_SEED + 1) % C.N_EPISODES]["reward"][0][1])

    def test_walker_w3f(self):
        self._check("w3f")

    def test_quadruped_2f_phys(self):
        self._check("quadruped_2f_phys")

    def test_quadruped_2f_mixed(self):
        self._check("quadruped_2f_mixed")


class MissingPayloadFailsClosed(unittest.TestCase):

    def _expect(self, bench, schedule, reward_schedule, message):
        fn = envchain.gym_env_fn(C.bench_config(bench), bench, mode="scheduled", schedule=schedule,
                                 reward_schedule=reward_schedule, worker_index=0, num_envs=1,
                                 base_seed=C.BASE_SEED)
        out = _run_child(_child_build_and_step, fn, 1)
        self.assertFalse(out["ok"], "a scheduled env without its payload was built and stepped")
        self.assertIn(message, out["error"])

    def test_walker_scheduled_without_physics_payload(self):
        self._expect("w3f", None, None, "no schedule payload")

    def test_quadruped_scheduled_without_physics_payload(self):
        self._expect("quadruped_2f_phys", None, None, "no schedule payload")

    def test_quadruped_mixed_scheduled_without_reward_payload(self):
        self._expect("quadruped_2f_mixed", C.load_payload("quadruped_2f_mixed", "ir_a"), None,
                     "no reward-label payload")


if __name__ == "__main__":
    unittest.main()
