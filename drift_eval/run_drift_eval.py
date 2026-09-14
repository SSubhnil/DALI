"""§8b gradual-shift battery for ONE DALI checkpoint (GPU + EGL; see drift_eval/README.md).

    python -m drift_eval.run_drift_eval --logdir <DALI logdir> --bench w3f \
        --schedule-dir drift_eval_results/schedules/w3f --results-root drift_eval_results \
        [--conditions A B C D E] [--n-episodes 64] [--num-envs 5]

Per condition it writes, in CWM's schema and ids (drift_eval/outputs.py, protocol.CONDITIONS):
raw/<row>__<cond>.json, traces/<row>__<cond>__trace.npz (+ .meta.json), and a row of
main_results.csv. Every output records the action mode, the generator path + sha256, the schedule
hashes and the CWM provenance fields.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

from drift_eval import protocol as P

# DALI's policy has no deterministic action path: every branch of Agent.policy samples
# (dreamerv3_compat/dreamerv3/agent.py:134-136 'eval', :161 'explore', :165 'train').
ACTION_MODES = {"sample": dict(policy_mode="eval",
                               source="dreamerv3_compat/dreamerv3/agent.py:134-136 "
                                      "(mode='eval' -> outs['action'].sample(seed=nj.rng()))")}


def _dali_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.dirname(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", required=True, help="DALI logdir with checkpoint.ckpt + config.yaml")
    ap.add_argument("--bench", required=True, choices=sorted(P.BENCHES))
    ap.add_argument("--schedule-dir", required=True,
                    help="directory the generator wrote for this bench (drift_eval.schedule)")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--conditions", nargs="+", default=["A", "B", "C", "D", "E"],
                    help="A B C D E, and F1 F8 F16 F64 F256 for the tau-dial")
    ap.add_argument("--n-episodes", type=int, default=64)
    ap.add_argument("--num-envs", type=int, default=P.DEFAULT_NUM_ENVS,
                    help="envs per batch; CWM's Evaluate.NumEnvs is 5 and the CRN plan must match")
    ap.add_argument("--row-id", default=None)
    ap.add_argument("--action-mode", default="sample", choices=sorted(ACTION_MODES))
    ap.add_argument("--generator", default=None, help="optional generator override (default: "
                    "DRIFT_GENERATOR_PATH, else the vendored drift_eval/vendor/drift_generator.py); "
                    "its sha256 must match VENDORED.md or DRIFT_GENERATOR_SHA256")
    args = ap.parse_args(argv)

    unknown = [c for c in args.conditions if c not in P.CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; choose from {sorted(P.CONDITIONS)}")

    P.cwm_root()
    import ruamel.yaml as ryaml
    import dreamerv3
    from dreamerv3 import embodied
    from envs import schedule_channel
    from drift_eval import envchain, outputs, schedule as S

    gen_mod, gen_path, gen_sha = P.load_generator(args.generator)
    values = P.load_values()
    prov = outputs.provenance_metadata(gen_path, gen_sha)
    bench = P.BENCHES[args.bench]

    with open(os.path.join(args.logdir, "config.yaml")) as f:
        raw_cfg = ryaml.YAML(typ="safe").load(f)
    config = embodied.Config(raw_cfg)
    envchain.check_bench_config(config, args.bench)
    trained_seed = int(config.seed)
    base_seed = P.eval_base_seed(trained_seed)
    config = config.update({"envs.amount": args.num_envs, "envs.parallel": "none"})
    row_id = args.row_id or f"dali_{args.bench}_s{trained_seed}"

    ckpt_path = embodied.Path(args.logdir) / "checkpoint.ckpt"
    assert ckpt_path.exists(), ckpt_path
    step_ckpt = embodied.Checkpoint()
    step_ckpt.step = embodied.Counter()
    step_ckpt.load(ckpt_path, keys=["step"])
    ckpt_step = int(step_ckpt._values["step"])

    payloads = {}
    for c in args.conditions:
        gc = P.CONDITIONS[c]["generator"]
        if gc is not None:
            payloads[c] = S.load_schedule(os.path.join(args.schedule_dir, P.schedule_filename(gc)),
                                          args.bench, gc, values, gen_mod.GENERATOR_VERSION)
    if len(payloads) > 1:
        schedule_channel.assert_crn_compatible(payloads)
    reward_payload, labels = None, None
    if bench["mixed"] and payloads:
        modes = list(config.env.carl.regime_b_reward_modes)
        reward_payload = S.load_reward_fixed(os.path.join(args.schedule_dir, P.REWARD_FIXED_FILE),
                                             modes, next(iter(payloads.values()))["n_episodes"],
                                             gen_mod.GENERATOR_VERSION)
        labels = outputs.label_plan(reward_payload, base_seed, args.n_episodes, args.num_envs)
        counts = {m: labels.count(m) for m in modes}
        if len(set(counts.values())) != 1:   # CWM run_eval_benchmark._resolve_schedule_modes
            raise SystemExit(f"the {args.n_episodes} retained episodes (num_envs={args.num_envs}) "
                             f"would run objectives {counts}, not balanced across {modes}")

    results_root = os.path.abspath(args.results_root)
    trace_root = os.path.join(results_root, "traces")
    git_sha = _dali_git_sha()
    am = ACTION_MODES[args.action_mode]
    wind_table = values[bench["values_keys"][3]] if 3 in bench["values_keys"] else None

    agent = None
    for c in args.conditions:
        cond = P.CONDITIONS[c]
        cid = cond["id"]
        payload = payloads.get(c)
        mode = "scheduled" if payload is not None else "bernoulli"
        rpay = reward_payload if (bench["mixed"] and payload is not None) else None
        wdir = os.path.join(trace_root, "_workers", f"{row_id}__{cid}")
        if os.path.isdir(wdir) and os.listdir(wdir):
            raise SystemExit(f"{wdir} already holds traces; refusing to mix runs")
        print(f"\n[cond] {c} {cid} mode={mode} "
              f"hash={(payload or {}).get('schedule_hash', '-')[:12]}", flush=True)

        envs = [envchain.build_embodied_env(config, args.bench, mode=mode, schedule=payload,
                                            reward_schedule=rpay, worker_index=i,
                                            num_envs=args.num_envs, base_seed=base_seed,
                                            trace_dir=wdir)
                for i in range(args.num_envs)]
        env = embodied.BatchEnv(envs, parallel=False)
        if agent is None:
            agent = dreamerv3.Agent(env.obs_space, env.act_space, ckpt_step, config)
            ag_ckpt = embodied.Checkpoint()
            ag_ckpt.agent = agent
            ag_ckpt.load(ckpt_path, keys=["agent"])

        per_worker_returns = {i: [] for i in range(args.num_envs)}
        per_worker_mse = {i: [] for i in range(args.num_envs)}
        hats, acts = {}, {}

        def on_step(tran, worker):
            if "reward_hat" in tran:
                hats.setdefault(worker, []).append(float(tran["reward_hat"]))
                acts.setdefault(worker, []).append(float(tran["reward"]))

        def on_episode(ep, worker):
            per_worker_returns[worker].append(float(np.asarray(ep["reward"], np.float64).sum()))
            h, a = hats.pop(worker, None), acts.pop(worker, None)
            per_worker_mse[worker].append(
                float(np.mean((np.asarray(h) - np.asarray(a)) ** 2))
                if h and a and len(h) == len(a) else float("nan"))

        driver = embodied.Driver(env)
        driver.on_step(on_step)
        driver.on_episode(on_episode)
        policy = lambda *a: agent.policy(*a, mode=am["policy_mode"])  # noqa: E731
        t0 = time.time()
        driver(policy, episodes=args.n_episodes)     # runs whole lockstep rounds; extras dropped below
        elapsed = time.time() - t0
        env.close()

        returns = outputs.select_plan_episodes(per_worker_returns, base_seed, args.n_episodes,
                                               args.num_envs)
        mses = outputs.select_plan_episodes(per_worker_mse, base_seed, args.n_episodes,
                                            args.num_envs)
        return_keys = outputs.plan_keys(base_seed, args.n_episodes, args.num_envs)
        extra = {
            "stack": "dali", "port_bench": args.bench, "condition_letter": c,
            "condition_label": cond["label"], "switch_mode": mode,
            "action_mode": args.action_mode, "policy_mode": am["policy_mode"],
            "action_mode_source": am["source"], "greedy": False,
            "ckpt_path": str(ckpt_path), "ckpt_step": ckpt_step, "trained_seed": trained_seed,
            "eval_base_seed": base_seed, "num_envs": args.num_envs,
            "episode_plan": "crn_episode_plan(base, n, num_envs): k = worker + ordinal*num_envs",
            "physics_schedule_hash": (payload or {}).get("schedule_hash"),
            "reward_schedule_hash": (rpay or {}).get("schedule_hash"),
            "stack_reference": ("CWM runner befabde (mixed drift)" if bench["mixed"]
                                else "CWM 45e8d5d runner"),
            "generator_version": getattr(gen_mod, "GENERATOR_VERSION", None),
            **prov, "elapsed_s": elapsed,
        }
        mse_arr = np.asarray(mses, dtype=float)
        mse_out = None if np.all(np.isnan(mse_arr)) else mse_arr
        outputs.write_condition_outputs(
            results_root, bench=args.bench, row_id=row_id, condition_id=cid, axis=cond["axis"],
            seed=trained_seed, git_sha=git_sha, returns=returns, return_keys=return_keys,
            mse=mse_out, traces=schedule_channel.collect_traces(wdir), payload=payload,
            reward_payload=rpay, labels=labels if rpay is not None else None,
            carl_factors=bench["carl_factors"], wind_table=wind_table, extra=extra,
            notes=(f"action_mode={args.action_mode};policy_mode={am['policy_mode']};"
                   + outputs.provenance_notes(prov)))
        print(f"  mean_return={np.mean(returns):.2f} n={len(returns)} elapsed={elapsed:.1f}s",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
