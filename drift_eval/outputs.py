"""Output writers in the SAME schema as CWM's runner, plus the fail-closed trace verification.

Schema of record (CWM 45e8d5d, plus the writer-only fields of CausalWorldModel @befabde):
  raw JSON   benchmark/run_eval_benchmark.py:112-138 @befabde (`episode_keys`, `metadata.paired_streams`)
  CSV row    benchmark/run_eval_benchmark.py:44-50 @45e8d5d, :46-52 @befabde (CSV_COLUMNS)
  trace cell benchmark/run_eval_benchmark.py:1218-1237 @befabde (`episode_worker`, `episode_ordinal`,
             `episode_return`; meta `paired_streams`, `row_join`, `unjoined_worker_traces`)
             -> benchmark/switch_recovery.py:745-757
Port-specific fields go INSIDE `metadata` (raw JSON) and the cell's `.meta.json` sidecar, so the
top-level keys and the npz arrays are exactly CWM's.
"""
from __future__ import annotations

import csv
import fcntl
import json
import os
import re
from datetime import datetime, timezone

import numpy as np

from drift_eval import protocol as P

RAW_JSON_KEYS = ("row_id", "condition_id", "returns", "reward_mse_per_episode", "n_episodes",
                 "episode_keys", "metadata")
RAW_JSON_METADATA_KEYS = ("seed", "env", "git_sha", "paired_streams", "eval_timestamp_utc")
CSV_COLUMNS = [
    "row_id", "condition_id", "method", "domain", "env", "experiment", "seed",
    "axis", "n_episodes", "mean_return", "std_return",
    "reward_mse_mean", "reward_mse_std",
    "raw_returns_path", "training_wandb_run_id", "eval_wandb_run_id",
    "eval_timestamp_utc", "git_sha", "notes",
]
CELL_KEYS = ("reward", "reward_mode_id", "factor_values", "factor_active", "lambda_sampled",
             "schedule_id", "episode_worker", "episode_ordinal", "episode_return")
CELL_META_KEYS = ("method", "env", "experiment", "seed", "axis", "condition_id", "row_id",
                  "lambda", "schedule_hash", "git_sha", "paired_streams", "row_join",
                  "unjoined_worker_traces")
ROW_JOIN = "by (episode_worker, episode_ordinal) to raw returns"   # @befabde :1231
# CWM's Evaluate.PairedStreams seeds the first random action of each episode (eval.py:431-432/:573 @befabde). DALI's
# eval has no such sampler: embodied Driver.reset starts from zero actions with the reset flag
# (dreamerv3_compat/dreamerv3/embodied/core/driver.py:26-29), masks actions of finished envs to zero
# (:51-53), FromGymnasium ignores the action on reset (embodied/envs/from_gymnasium.py:75-80), and the
# only act_space.sample is DALI's --random_policy path (contextual_mbrl/dreamer/eval.py:81-85), unused
# here. So the value is null, never true/false, with the reason written beside it.
DALI_PAIRED_STREAMS = None
DALI_PAIRED_STREAMS_REASON = "not_applicable: port has no first-action sampler"

_TAG_RE = re.compile(r"^w(\d+)(lf|rf)$")

PROVENANCE_KEYS = ("cwm_root", "cwm_commit", "cwm_dirty_count", *P.CWM_PRESET_YAMLS,
                   "generator_path", "generator_sha256")


def provenance_metadata(generator_path, generator_sha256):
    """Fields written into EVERY output (raw JSON metadata, trace-cell sidecar, CSV notes,
    GENERATOR_PROVENANCE.json): CWM root/commit/dirty count, the three preset yaml sha256s, the
    generator path/sha256, and the full CWM source sha256 map."""
    meta = dict(P.cwm_provenance())
    meta.update(generator_path=generator_path, generator_sha256=generator_sha256,
                sources=P.source_provenance(generator_path))
    return meta


def provenance_notes(meta):
    """Compact `k=v;` form of the provenance fields for the CSV `notes` column."""
    return ";".join(f"{k}={meta[k]}" for k in PROVENANCE_KEYS)


def write_raw_json(raw_dir, row_id, condition_id, returns, mse, *, seed, env, git_sha,
                   extra_metadata, episode_keys=None, paired_streams=None):
    os.makedirs(raw_dir, exist_ok=True)
    filename = f"{row_id}__{condition_id}.json"
    payload = {
        "row_id": row_id,
        "condition_id": condition_id,
        "returns": [float(x) for x in returns],
        "reward_mse_per_episode": ([float(x) for x in mse] if mse is not None else None),
        "n_episodes": len(returns),
        "episode_keys": ([[int(w), int(j)] for w, j in episode_keys]
                         if episode_keys is not None else None),
        "metadata": {**extra_metadata, "seed": seed, "env": env, "git_sha": git_sha,
                     "paired_streams": paired_streams,
                     "paired_streams_reason": DALI_PAIRED_STREAMS_REASON,
                     "eval_timestamp_utc": datetime.now(timezone.utc).isoformat()},
    }
    with open(os.path.join(raw_dir, filename), "w") as f:
        json.dump(payload, f)
    return os.path.join("raw", filename)


def append_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    needs_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if needs_header:
                w.writeheader()
            w.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def join_traces_to_returns(traces, return_keys, suffix, cid="?"):
    """VERBATIM mirror of CWM `_join_traces_to_returns` (CausalWorldModel @befabde,
    benchmark/run_eval_benchmark.py:659-688). Not imported: that module imports torch, which the DALI
    venv does not have, and CAUSAL_WORLD_MODEL_ROOT @45e8d5d predates it.

    Order worker traces to match raw returns BY (worker, episode ordinal), never by position.
    Returns (joined, n_unjoined).
    """
    if return_keys is None:
        raise SystemExit(f"[D1] condition {cid!r}: eval_episodes returned no per-episode keys; "
                         f"traces cannot be joined to returns.")
    keys = [(int(w), int(j)) for w, j in return_keys]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"[D1] condition {cid!r}: duplicate episode keys in the returns {keys}.")
    pattern = re.compile(rf"^w(\d+){re.escape(suffix)}$")
    by_key = {}
    for t in traces:
        m = pattern.match(str(t.get("worker_tag", "")))
        if m is None or "episode_ordinal" not in t:
            raise SystemExit(f"[D1] condition {cid!r}: trace without a (worker, ordinal) key "
                             f"(worker_tag={t.get('worker_tag')!r}).")
        key = (int(m.group(1)), int(t["episode_ordinal"]))
        if key in by_key:
            raise SystemExit(f"[D1] condition {cid!r}: two traces for episode {key}.")
        by_key[key] = t
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise SystemExit(f"[D1] condition {cid!r}: no trace for returned episodes {missing[:8]}.")
    return [by_key[k] for k in keys], len(by_key) - len(keys)


def order_and_verify(traces, *, return_keys, suffix, payload=None, carl_factors=(),
                     driver_returns=None, reward_payload=None, wind_table=None, cid="?"):
    """Join worker traces to the returns by (worker, ordinal) with CWM's join (traces filtered by
    suffix first, as run_eval_benchmark.py:1188-1189 @befabde does), then REFUSE the condition
    unless every joined trace has T = 500 decision rows 1..T; under a schedule every active column
    equals the schedule after CWM's clamp, exactly (float32) and carries its hash; the fixed objective
    holds on mixed benches; and (if given) the driver's return matches the trace's reward sum.
    Returns (joined traces in return order, number of unjoined traces)."""
    worker_traces = [t for t in traces if str(t.get("worker_tag", "")).endswith(suffix)]
    ordered, unjoined = join_traces_to_returns(worker_traces, return_keys, suffix, cid)
    if payload is not None and "wind_x" in carl_factors and wind_table is None:
        raise ValueError("[drift_eval] a wind factor is scheduled: pass wind_table so the check "
                         "applies CWM's wind clamp")
    T = P.EPISODE_T
    for k, tr in enumerate(ordered):
        if len(tr["reward"]) != T or not np.array_equal(tr["decision_step"],
                                                         np.arange(1, T + 1)):
            raise RuntimeError(f"[drift_eval] episode {k} has {len(tr['reward'])} rows, not 1..{T}")
        if payload is not None:
            if str(tr["schedule_hash"]) != payload["schedule_hash"]:
                raise RuntimeError(f"[drift_eval] episode {k} ran schedule "
                                   f"{str(tr['schedule_hash'])[:12]}, not {payload['schedule_hash'][:12]}")
            ep = payload["episodes"][int(tr["schedule_episode_index"])]
            for name in carl_factors:
                fid = P.CARL_NAME_TO_FACTOR_ID[name]
                want = np.asarray([P.our_clip(name, v, wind_table)
                                   for _, v in ep[f"physics_f{fid}"][1:]], dtype=np.float32)
                if not np.array_equal(tr["factor_values"][:, fid], want):
                    raise RuntimeError(f"[drift_eval] episode {k} factor {name}: applied path "
                                       f"differs from the schedule")
        if reward_payload is not None:
            label = reward_payload["episodes"][int(tr["reward_schedule_episode_index"])]["reward"][0][1]
            ids = np.unique(tr["reward_mode_id"])
            if str(tr["objective"]) != label or len(ids) != 1:
                raise RuntimeError(f"[drift_eval] episode {k}: objective {tr['objective']} / mode ids "
                                   f"{ids.tolist()} != fixed label {label}")
        if driver_returns is not None:
            r = float(np.asarray(tr["reward"], dtype=np.float64).sum())
            if abs(r - driver_returns[k]) > 1e-3 * max(1.0, abs(r)):
                raise RuntimeError(f"[drift_eval] episode {k}: driver return {driver_returns[k]} "
                                   f"!= trace reward sum {r}; traces and episodes are misaligned")
    return ordered, unjoined


def plan_keys(base_seed, n_episodes, num_envs):
    """(worker, ordinal) of each retained return, in CWM's order (eval.py:701 @befabde `_raw_episode_keys`)."""
    from envs.schedule_channel import crn_episode_plan
    return [(p["worker"], p["episode_in_worker"])
            for p in crn_episode_plan(base_seed, n_episodes, num_envs)]


def select_plan_episodes(per_worker, base_seed, n_episodes, num_envs):
    """Retained episodes in CWM's order: crn_episode_plan(base, n, num_envs), i.e. episode k is worker
    k % num_envs, ordinal k // num_envs; completions past the n-th are dropped (eval.py:778 @befabde)."""
    from envs.schedule_channel import crn_episode_plan
    return [per_worker[p["worker"]][p["episode_in_worker"]]
            for p in crn_episode_plan(base_seed, n_episodes, num_envs)]


def label_plan(reward_payload, base_seed, n_episodes, num_envs):
    """CWM run_eval_benchmark._drift_label_plan (:476 @befabde): labels[(worker_seed + episode_in_worker) % n]."""
    from envs.schedule_channel import crn_episode_plan
    labels = [ep["reward"][0][1] for ep in reward_payload["episodes"]]
    return [labels[(p["worker_seed"] + p["episode_in_worker"]) % len(labels)]
            for p in crn_episode_plan(base_seed, n_episodes, num_envs)]


def cell_meta(*, bench, condition_id, axis, row_id, seed, git_sha, physics_payload,
              reward_payload, labels=None, extra=None, paired_streams=None, unjoined=0):
    """Trace-cell sidecar as CWM run_eval_benchmark.py:1224-1237 @befabde writes it: on a reward-wrapper
    bench `schedule_hash` is the REWARD schedule's; a mixed drift cell adds physics_schedule_hash and
    reward_label_plan. CWM's keys override any same-named extra."""
    b = P.BENCHES[bench]
    sched = (reward_payload if b["mixed"] else physics_payload) or {}
    meta = {"method": "dali", "env": b["env"], "experiment": b["generator_bench"], "seed": seed,
            "axis": axis, "condition_id": condition_id, "row_id": row_id, "lambda": None,
            "schedule_hash": sched.get("schedule_hash"), "git_sha": git_sha,
            "paired_streams": paired_streams,
            "paired_streams_reason": DALI_PAIRED_STREAMS_REASON, "row_join": ROW_JOIN,
            "unjoined_worker_traces": int(unjoined)}
    if b["mixed"] and physics_payload is not None:
        meta["physics_schedule_hash"] = physics_payload.get("schedule_hash")
        meta["reward_label_plan"] = labels
    return {**(extra or {}), **meta}


def csv_row(*, bench, condition_id, axis, row_id, seed, returns, mse, raw_rel, git_sha, notes):
    """CSV row as CWM run_eval_benchmark.py:1129-1167 @befabde: mean/std over episodes with np.std ddof 0
    (eval.py:827 @befabde), experiment = CWM bench name, env = CWM env id."""
    b = P.BENCHES[bench]
    r = np.asarray(returns, dtype=np.float64)
    m = None if mse is None else np.asarray(mse, dtype=np.float64)
    return {"row_id": row_id, "condition_id": condition_id, "method": "dali", "domain": "dmc",
            "env": b["env"], "experiment": b["generator_bench"], "seed": seed, "axis": axis,
            "n_episodes": len(r), "mean_return": float(np.mean(r)), "std_return": float(np.std(r)),
            "reward_mse_mean": float(np.mean(m)) if m is not None and len(m) else float("nan"),
            "reward_mse_std": float(np.std(m)) if m is not None and len(m) else float("nan"),
            "raw_returns_path": raw_rel, "training_wandb_run_id": "", "eval_wandb_run_id": "",
            "eval_timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha,
            "notes": notes}


def notes_with_paired_streams(notes, paired_streams):
    """CSV notes tag for paired_streams. CWM `_notes_with_paired_streams` (run_eval_benchmark.py:615-629
    @befabde) writes `paired_streams=<bool>`, or `paired_streams=null (<reason>)` for a port. This port
    writes `paired_streams=null; paired_streams_reason=<reason>`: same value and reason, different
    notes text (the JSON/meta fields match CWM's `_paired_streams_from_record`). Never coerced to False."""
    if paired_streams is None:
        tag = f"paired_streams=null; paired_streams_reason={DALI_PAIRED_STREAMS_REASON}"
    else:
        tag = f"paired_streams={bool(paired_streams)}"
    return f"{notes}; {tag}" if notes else tag


def build_cell(ordered_traces, return_keys, returns, lambda_value=None):
    """CWM cell arrays; row i IS returns[i] (episode_worker/episode_ordinal = its join key)."""
    return {
        "reward": np.stack([t["reward"] for t in ordered_traces]),
        "reward_mode_id": np.stack([t["reward_mode_id"] for t in ordered_traces]),
        "factor_values": np.stack([t["factor_values"] for t in ordered_traces]),
        "factor_active": ordered_traces[0]["factor_active"],
        "lambda_sampled": np.array([lambda_value if lambda_value is not None else float("nan")]
                                   * len(ordered_traces), dtype=np.float32),
        "schedule_id": np.arange(len(ordered_traces), dtype=np.int32),
        "episode_worker": np.array([int(k[0]) for k in return_keys], dtype=np.int32),
        "episode_ordinal": np.array([int(k[1]) for k in return_keys], dtype=np.int32),
        "episode_return": np.asarray(returns, dtype=np.float32),
    }


def write_condition_outputs(results_root, *, bench, row_id, condition_id, axis, seed, git_sha,
                            returns, return_keys, mse, traces, payload, reward_payload, labels,
                            carl_factors, wind_table, extra, notes):
    """The port's writer path for one condition: keyed join + verification, raw JSON, trace cell,
    CSV row. Returns (raw JSON path, cell path)."""
    suffix = "rf" if P.BENCHES[bench]["mixed"] else "lf"
    ordered, unjoined = order_and_verify(
        traces, return_keys=return_keys, suffix=suffix, payload=payload,
        reward_payload=reward_payload, carl_factors=carl_factors, driver_returns=returns,
        wind_table=wind_table, cid=condition_id)
    meta = cell_meta(bench=bench, condition_id=condition_id, axis=axis, row_id=row_id, seed=seed,
                     git_sha=git_sha, physics_payload=payload, reward_payload=reward_payload,
                     labels=labels, extra=extra, paired_streams=DALI_PAIRED_STREAMS,
                     unjoined=unjoined)
    raw_rel = write_raw_json(os.path.join(results_root, "raw"), row_id, condition_id, returns, mse,
                             seed=seed, env=P.BENCHES[bench]["env"], git_sha=git_sha,
                             extra_metadata=meta, episode_keys=return_keys,
                             paired_streams=DALI_PAIRED_STREAMS)
    cell_path = write_cell(os.path.join(results_root, "traces",
                                        f"{row_id}__{condition_id}__trace.npz"),
                           build_cell(ordered, return_keys, returns), meta)
    append_csv_row(os.path.join(results_root, "main_results.csv"), csv_row(
        bench=bench, condition_id=condition_id, axis=axis, row_id=row_id, seed=seed,
        returns=returns, mse=mse, raw_rel=raw_rel, git_sha=git_sha,
        notes=notes_with_paired_streams(notes, DALI_PAIRED_STREAMS)))
    return os.path.join(results_root, raw_rel), cell_path


def write_cell(path, cell, meta):
    P.cwm_root()
    from benchmark.switch_recovery import write_trace_cell
    return write_trace_cell(path, cell, meta)
