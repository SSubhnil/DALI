"""Schedule files: produce them with OUR generator (by path), and load them with fail-closed checks.

    python -m drift_eval.schedule --bench w3f --out-root drift_eval_results/schedules [--episodes 64]

calls the loaded generator's own `main()` CLI (never a re-implementation) with the bench's
--factors/--domain and the CWM value table, then writes GENERATOR_PROVENANCE.json beside the files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from drift_eval import protocol as P

_HASH_FIELDS = ("generator_version", "condition", "processes", "lambda", "T", "episodes")


def recompute_schedule_hash(payload: dict) -> str:
    """The content hash as drift_generator.build_condition computes it (its :280-283 in the vendored befabde file)."""
    return hashlib.sha256(json.dumps({k: payload[k] for k in _HASH_FIELDS},
                                     sort_keys=True).encode()).hexdigest()


def load_schedule(path: str, bench: str, generator_condition: str, values: dict,
                  generator_version: str) -> dict:
    """Load one generator payload and refuse it unless it is exactly what this bench/condition
    needs: intact hash, same generator version, same condition, the bench's processes with OUR
    train values, T = 500, dense decision-step events for every episode."""
    with open(path) as f:
        payload = json.load(f)
    problems = []
    if recompute_schedule_hash(payload) != payload.get("schedule_hash"):
        problems.append("schedule_hash does not match content")
    if payload.get("generator_version") != generator_version:
        problems.append(f"generator_version {payload.get('generator_version')!r} != loaded "
                        f"generator {generator_version!r}")
    want_cond = "tau" if generator_condition.startswith("tau_") else generator_condition
    if payload.get("condition") != want_cond:
        problems.append(f"condition {payload.get('condition')!r} != {want_cond!r}")
    want_proc = P.expected_processes(bench, values)
    got_proc = {k: [float(v) for v in vs] for k, vs in payload.get("processes", {}).items()}
    if got_proc != want_proc:
        problems.append(f"processes {got_proc} != bench {bench} expects {want_proc}")
    T = int(payload.get("T", -1))
    if T != P.EPISODE_T:
        problems.append(f"T={T} != {P.EPISODE_T}")
    eps = payload.get("episodes", [])
    if not eps or len(eps) != int(payload.get("n_episodes", -1)):
        problems.append("episodes empty or n_episodes mismatch")
    for j, ep in enumerate(eps):
        if set(ep) != set(want_proc):
            problems.append(f"episode {j} keys {sorted(ep)} != {sorted(want_proc)}")
            break
        for key, seq in ep.items():
            if [int(s) for s, _ in seq] != list(range(T + 1)):
                problems.append(f"episode {j} {key} is not dense 0..{T}")
                break
    if problems:
        raise ValueError(f"[drift_eval] refusing schedule {path}: " + "; ".join(problems))
    return payload


def load_reward_fixed(path: str, modes, n_episodes: int, generator_version: str) -> dict:
    """The mixed-bench reward-label payload (generator befabde `build_reward_fixed_schedule` :352-388):
    one label per episode at step 0, balanced, same episode grid as the physics payloads."""
    with open(path) as f:
        payload = json.load(f)
    problems = []
    if recompute_schedule_hash(payload) != payload.get("schedule_hash"):
        problems.append("schedule_hash does not match content")
    if payload.get("generator_version") != generator_version:
        problems.append("generator_version mismatch")
    if payload.get("condition") != "reward_fixed":
        problems.append(f"condition {payload.get('condition')!r} != 'reward_fixed'")
    if list(payload.get("processes", {}).get("reward", [])) != list(modes):
        problems.append(f"reward modes {payload.get('processes')} != checkpoint's {list(modes)}")
    eps = payload.get("episodes", [])
    if int(payload.get("n_episodes", -1)) != n_episodes or len(eps) != n_episodes:
        problems.append(f"n_episodes {payload.get('n_episodes')} != physics grid {n_episodes}")
    if int(payload.get("T", -1)) != P.EPISODE_T:
        problems.append("T mismatch")
    for j, ep in enumerate(eps):
        seq = ep.get("reward") or []
        if len(seq) != 1 or int(seq[0][0]) != 0 or seq[0][1] not in modes:
            problems.append(f"episode {j} reward entry {seq} is not one step-0 label")
            break
    if problems:
        raise ValueError(f"[drift_eval] refusing reward schedule {path}: " + "; ".join(problems))
    return payload


def make_schedules(bench: str, out_root: str, episodes: int, generator_file: str | None = None,
                   taus: str = "1,8,16,64,256") -> str:
    mod, gpath, gsha = P.load_generator(generator_file)
    b = P.BENCHES[bench]
    out_dir = os.path.join(out_root, bench)
    argv = ["drift_generator", "--out-dir", out_dir, "--values", P.values_yaml_path(),
            "--bench", b["generator_bench"],
            "--episodes", str(episodes), "--T", str(P.EPISODE_T), "--taus", taus]
    old = sys.argv
    try:
        sys.argv = argv
        rc = mod.main()
    finally:
        sys.argv = old
    if rc not in (0, None):
        raise SystemExit(f"generator exited {rc}")
    with open(os.path.join(out_dir, "GENERATOR_PROVENANCE.json"), "w") as f:
        json.dump({"bench": bench, "argv": argv[1:], "generator_path": gpath,
                   "generator_sha256": gsha, "generator_version": mod.GENERATOR_VERSION,
                   **P.cwm_provenance(), "sources": P.source_provenance(gpath)}, f, indent=2)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, choices=sorted(P.BENCHES))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--generator", default=None, help="optional override of the vendored generator "
                    "(sha256 must match VENDORED.md or DRIFT_GENERATOR_SHA256)")
    ap.add_argument("--taus", default="1,8,16,64,256")
    a = ap.parse_args()
    print(make_schedules(a.bench, a.out_root, a.episodes, a.generator, a.taus))


if __name__ == "__main__":
    main()
