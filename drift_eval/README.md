# drift_eval — §8b gradual-shift evaluation of DALI checkpoints

Runs the CausalWorldModel (CWM) §8b gradual-shift conditions on DALI's terminal checkpoints, using
CWM's schedule generator (vendored), value tables and regime-B wrappers, and writes CWM's output schema.
Protocol of record: CWM `docs/ICLR27/ICLR_PAPER_PLAN.md` §8b (design) and §8c (logging).

**Quadruped benches (`quadruped_2f_phys`, `quadruped_2f_mixed`) are a mirror of the walker design
ruled by trajd-advisor 2026-09-14, pending PI ratification. They are not in §8b.**

**Requirement.** The port needs a CausalWorldModel checkout at commit 45e8d5d or later, given by
`CAUSAL_WORLD_MODEL_ROOT` (default `/home/jovyan/m10_code/45e8d5d`). The bench presets, value
table and regime-B wrappers are read from it, not vendored. The loader records in every output
file (raw JSON `metadata`, trace-cell `.meta.json`, CSV `notes`, `GENERATOR_PROVENANCE.json`):
`cwm_root`, `cwm_commit` (`git rev-parse HEAD`, or "not a git checkout"), `cwm_dirty_count`
(lines of `git status --porcelain`), `cwm_sha256_experiments_dmc_yaml`,
`cwm_sha256_configure_yaml`, `cwm_sha256_latent_factor_values_yaml`, `generator_path` and
`generator_sha256`.

**Status.** Built and tested on CPU only (physics level, no rendering). The GPU path (checkpoint load,
JAX policy, EGL rendering, `run_drift_eval.py` end to end) has **not** been executed. Run the GPU
smoke below before any battery.

## Conditions and ids (CWM's own)

| letter (CLI) | CWM condition id | CWM axis | switching | source |
|---|---|---|---|---|
| A | `phys_train` | physics | Bernoulli regime B (λ ~ U[0.003, 0.008] per factor, per episode) | the training wrapper, CWM-parity fixes below |
| B | `drift_ir_a` | drift | IR-A, matched abrupt | generator `ir_a.json` |
| C | `drift_ir_s` | drift | IR-S, gradual sine | generator `ir_s.json` |
| D | `drift_h_a` | drift | H-A, hard matched abrupt | generator `h_a.json` |
| E | `drift_h_s` | drift | H-S, hard gradual sine | generator `h_s.json` |
| F1…F256 | `drift_tau_<k>` | drift | τ-dial (optional) | generator `tau_<k>.json` |

The ids and axes are CWM's (CausalWorldModel @befabde `benchmark/eval_conditions.yaml`:
`phys_train` :28/:47/:77, drift ids :143-163). Other fields written as CWM writes them:

| field | value written (= CWM) | CWM source |
|---|---|---|
| `experiment` | CWM bench name: `walker_3f_phys`, `walker_2f_phys`, `walker_2f_mixed`, `quadruped_2f_phys`, `quadruped_2f_mixed` (port bench `w3f`, `w2fphys`, `w2fmix`, same, same) | eval_manifest.yaml |
| `env` / `domain` | `dm_walker_walk` or `dm_quadruped_walk` / `dmc` | eval_manifest.yaml |
| `reward_mode_id` (trace) | index in the SORTED domain reward registry: walker `walk_backward` 5, `walk_forward` 6; quadruped `go_north` 3, `go_south` 4 | `envs/dmc_reward_factors.py:359/:429` |
| cell `schedule_hash` | mixed bench: the REWARD schedule's hash, plus `physics_schedule_hash` and `reward_label_plan`; physics bench: the physics schedule's | `run_eval_benchmark.py:1177/:1224-1237` @befabde |
| `std_return` | `np.std`, ddof 0 | `eval.py:827` @befabde |
| raw JSON `episode_keys`, `metadata.paired_streams` | (worker, ordinal) of each return; `paired_streams: null` + `paired_streams_reason: "not_applicable: port has no first-action sampler"` (CWM's PairedStreams seeds the first random action, `eval.py:431-432/:573` @befabde; DALI's driver starts from zero actions, `embodied/core/driver.py:26-29`, `:51-53`) — same in cell meta and CSV notes | `run_eval_benchmark.py:126/:132` @befabde |
| cell `episode_worker`, `episode_ordinal`, `episode_return`; meta `paired_streams`, `row_join`, `unjoined_worker_traces` | row i is returns[i], joined by (worker, ordinal) | `run_eval_benchmark.py:1218-1232` @befabde |
| envs per batch | 5; retained episodes = `crn_episode_plan(base, n, 5)` (64 episodes: 13 lockstep rounds, the 65th completion is dropped) | `eval.py:778` @befabde, `schedule_channel.crn_episode_plan` |

Mixed benches (`w2fmix`, `quadruped_2f_mixed`) follow the generator's output. Gravity follows the
physics schedule. The reward objective is fixed within each episode, and its label comes from
`reward_fixed.json` (balanced across episodes). The CWM reference for the mixed benches is the mixed-drift runner (CausalWorldModel @befabde). At 45e8d5d, CWM `run_eval_benchmark.py:799` refused drift on
reward-factor families.

## Benches and checkpoints (checked on the devbox 2026-09-14)

| bench | DALI `regime_b_factors` | factors (CWM value-table keys) | seeds on disk | metrics.jsonl last step |
|---|---|---|---|---|
| `w3f` | K3 | actuator_strength, gravity, wind_walker | 1–3 `baseline_ckpts/dali_parity/dmc_walker_3f/seed{1,2,3}`; 4–5 `DALI/logs/retrain_950k/walker_3f_s{4,5}` | 949,896 (all) |
| `w2fphys` | K2 | actuator_strength, gravity | 1–3 `baseline_ckpts/dali_parity/dmc_walker_2f/seed*`; 4–5 `retrain_950k/walker_2f_s{4,5}` | 949,896 (all) |
| `w2fmix` | K3_mixed | gravity; reward [walk_forward, walk_backward] | 1–5 `retrain_950k/walker_mixed_s{1..5}` | s1 949,917; s2–s5 949,896 |
| `quadruped_2f_phys` | K2 | joint_damping (factor id 0), gravity | 1–3 `baseline_ckpts/dali_parity/dmc_quadruped_2f/seed*`; 4–5 `retrain_950k/quad_2f_s{4,5}` | 949,896 (all) |
| `quadruped_2f_mixed` | K3_mixed (gravity only) | gravity; reward [go_north, go_south] | 1–3 `baseline_ckpts/dali_parity/dmc_quadruped_2f_mixed/seed*`; 4–5 `retrain_950k/quad_mixed_s{4,5}` | s2 949,897; others 949,896 |

The full paths are in `drift_eval.yaml`. Seeds and factor sets were read from each logdir's
`config.yaml`; steps are the last `step` in `metrics.jsonl`. The runner records the checkpoint's own
step counter. Two things were not checked: HuggingFace (offline box), and whether CWM's
`eval_manifest.yaml` rows `retrain_950k/walker_3f_s1..3` etc. exist elsewhere. Those directories are
not under `DALI/logs/retrain_950k` on this box; the seed 1–3 copies are under `baseline_ckpts/dali_parity`.

## How to run (GPU node, hardware EGL)

```bash
# 1. GPU smoke first: one seed, 4 episodes, two conditions. Record wall-clock and VRAM from it.
CONDITIONS="A B" N_EPISODES=10 bash drift_eval/launch_drift_eval.sh 0 w3f 1
# 2. Battery for one bench (every seed in drift_eval.yaml, or list seeds after the bench)
bash drift_eval/launch_drift_eval.sh 0 w3f
```

`launch_drift_eval.sh` generates the bench's schedules once, with the configured generator and 64
episodes. It then runs `python -m drift_eval.run_drift_eval` per checkpoint.

- **Compute:** one GPU per process for the JAX policy plus `NUM_ENVS=5` in-process EGL render
  envs, and CPU for 5 MuJoCo envs.
- **Wall-clock and VRAM: to be measured on the GPU smoke.** They could not be measured on the CPU box.
- **Budget:** 64 episodes × 5 conditions × 5 seeds × 5 benches = 8,000 episodes of 500 decisions.

## Outputs (CWM schema, under `drift_eval_results/`, git-ignored)

- `raw/<row>__<cond>.json`: CWM `run_eval_benchmark.py:112-138` @befabde. Top-level keys are exactly
  `row_id, condition_id, returns, reward_mse_per_episode, n_episodes, metadata`. Port fields sit
  inside `metadata`: `action_mode`, `generator_path`, `generator_sha256`, `schedule_hash`,
  `reward_schedule_hash`, `ckpt_step`, CWM source sha256s, and `stack_reference`.
- `traces/<row>__<cond>__trace.npz` + `.meta.json`: CWM trace cell (`run_eval_benchmark.py:1218-1237` @befabde,
  written by CWM `switch_recovery.write_trace_cell`).
  - Arrays: `reward[N,500]`, `reward_mode_id[N,500]`, `factor_values[N,500,4]`, `factor_active[4]`,
    `lambda_sampled[N]`, `schedule_id[N]`.
  - `factor_values` is in OUR units (gravity multiplier) and canonical column order
    (actuator_strength/joint_damping, gravity, friction, wind), NaN where inactive.
  - Row i is decision step i+1.
- `main_results.csv`: CWM `CSV_COLUMNS`; the action mode and hashes go in `notes`.
- Episodes are ordered by the CRN index k = worker + ordinal·num_envs.

## What guarantees the schedule was applied (fail closed)

1. The payload is a constructor argument of the env wrapper (`envchain.py`), never a module global.
   `gym_env_fn`/`embodied_env_fn` are picklable partials that carry it into whichever process
   builds, and therefore steps, the env.
2. `mode='scheduled'` without a physics payload raises at construction. On a mixed bench, a missing
   reward-label payload also raises.
3. After every reset and every step, the wrapper checks its factor state AND the MuJoCo model
   (gravity, actuator gear, dof damping) against the schedule, exactly, in the stepping process.
4. Scheduled values are clamped exactly as CWM's applier clamps them (same bounds, in our units,
   before conversion), so the applied factor vector matches CWM's bit for bit. DALI's own wrapper
   clip is bypassed under a schedule. `schedule_clipped_entries` in each worker trace counts the
   clamped entries.
5. A PhysicsError retry during reset raises. DALI's retry would replace the episode with an unpaired one.
6. After each condition, the runner checks that every one of the N traces has 500 rows and the
   condition's schedule hash, that its factor columns equal the schedule, that the driver return
   equals the trace reward sum, and (mixed benches) that the objective equals the label. Any
   mismatch aborts the condition.

## Units and action mode

- **Gravity conversion (the only place):** `drift_eval/protocol.py`,
  `GRAVITY_MS2_PER_MULTIPLIER = 9.81`, `canonical_to_carl` / `carl_to_canonical`. Schedules and
  traces use multipliers; DALI's CARL wrapper applies m/s².
- **Action mode:** sampled. DALI's `Agent.policy` samples in every mode
  (`dreamerv3_compat/dreamerv3/agent.py:134-136` eval, `:161` explore, `:165` train). The runner
  uses `mode="eval"`, as DALI's `contextual_mbrl/dreamer/eval.py:138` does. **No greedy flag was
  added.** A mean-action path is not a one-line change: it needs a new mode branch in the jitted
  policy (`:134` condition and `:136` `.sample(...)` → `.mode()`), a new jit trace, and a
  `tfd.Independent(tfd.Normal(tanh(out), std)).mode()` semantics check (`nets.py:795-801`).
  None of that can be exercised without a GPU and a checkpoint. Every output records
  `action_mode: sample`.

## Footing: every axis that differs from our battery

| axis | CWM battery (45e8d5d) | this DALI port | matched? |
|---|---|---|---|
| episodes / condition | 64 | 64 | yes |
| trained seeds | 1–3 (§8b) | 1–5 on disk; launcher runs all listed unless seeds are given | **no**, unless restricted to 1–3 |
| checkpoint | terminal `_final` at 950k | `checkpoint.ckpt`, metrics last step 949,896–949,917 | approximately; step recorded |
| action selection | greedy [inferred from CWM `docs/trajd/a2c_reads_2026-09-11/DALI_ENV_PARITY_2026-09-13.md` §3; not re-read in the §8b runner] | **sampled only. No greedy flag: it needs a new branch in the jitted policy (`agent.py:134-136`), which cannot be tested on CPU** | **no** |
| policy noise across conditions | none (greedy) | JAX RNG; not common across conditions | **no** |
| simulator / env stack | CWM `envs/my_dmc.DMControl` + `DMControlLatentFactors` | CARL `CARLDmc*Env` (carl-bench 1.1.1, gymnasium 1.1.1 against a declared <1.0) + CWM `carl_intra_episode.py` | **no** (config parity only; trajectory parity not checked here) |
| renderer | EGL | EGL | yes (different render paths) |
| image | 64×64 | 64×64, BILINEAR `ResizeImage` (DALI `envs.py:655-662`) | resize method on our side not checked |
| action repeat / episode length | 2 / 500 decisions | 2 / 500 decisions (verified on CPU) | yes |
| factor application timing | factor written, then `physics.step` with no `mj_forward` | same (the CARL wrapper's post-switch `forward()` is suppressed during `step`) | yes, bitwise (verifier, synced 14/14); see the note below the table |
| Bernoulli value tables | multipliers, gravity `v * (-9.81)` | CWM's table in CARL units, gravity `v * 9.81` (not CARL's stored 3.924…) | yes, bitwise |
| Bernoulli draw order | per factor in factor-id order (`dmc_latent_factors.py:367/:407`) | names re-sorted by factor id (CARL sorts alphabetically: quadruped gravity before joint_damping) | yes |
| reward RNG seed | worker seed (`eval.py:118`) | worker seed (DALI native: seed + 1000) | yes |
| gravity unit | multiplier of 9.81 | m/s², converted in `protocol.py` | converted, bitwise (`v*(-9.81) == -(v*9.81)`) |
| clip under schedule | `float(np.clip(v, 0.1, 10))` in our units, then `* (-9.81)` for gravity; wind ±max\|wind table\| = ±120 N walker (`dmc_latent_factors.py:257-274`) | the same clamp, same units, same order (`protocol.our_clip`), then the one unit conversion; the quadruped H-S joint_damping peak 10.000000000000002 becomes 10.0. **DALI's native gravity clip (39.24 m/s² = 4×) is bypassed under a schedule**; it would cut the H-S/H-A gravity path (peak 5.126×) | yes, bit for bit (tested) |
| clip under A (Bernoulli) | [0.1, 10]× | DALI native (gravity [0.98, 39.24] m/s², actuator/damping [0.05, 15]) | equal on the train values |
| initial-state seeding | one task RNG per worker, `RandomState(base + i)` (`envs/my_dmc.py:43-50`, `eval.py:108`); episode j = its j-th draw | the same derivation: one `RandomState(worker seed)` handed to every rebuilt CARL task (`protocol.worker_task_rng`) | yes: step-0 qpos equal bitwise (tested, w3f + quadruped) |

DALI reset seeding replaced by CausalWorldModel's per-(worker, ordinal) derivation so episodes are paired across models; DALI's original `[worker_seed, ordinal+1]` seeding is not reproduced.
| env seed | `Seed*1000+7`, worker i = base+i (CWM `eval.py:48`, `schedule_channel.crn_episode_plan`) | same formula on DALI's trained seed | formula yes; streams differ |
| envs per batch | `Evaluate.NumEnvs` = 5 | 5; same retained-episode plan | yes |
| schedule episode index | (worker seed + ordinal) mod n | same | yes |
| reward-label index (mixed) | (worker seed + ordinal) mod n (`dmc_reward_factors.py:376/:452`) | same, from `reward_fixed.json` | yes (CWM mixed-drift runner befabde) |
| condition A switching | Bernoulli, λ ~ U[0.003, 0.008] | same CWM wrapper code, via DALI's factory | yes |
| PhysicsError | raises under instrumentation | raises on reset retry; a mid-step error propagates, no `RestartOnException` | yes (both fail closed) |
| quadruped design | generator befabde mirror | same generator file | mirror pending PI |

Factor application timing matched to CausalWorldModel `envs/dmc_latent_factors.py:440/:451` (factor written, then `physics.step` with no `mj_forward`: gear, gravity and damping lag one physics substep of each decision). The DALI/cRSSM checkpoints were trained through CausalWorldModel's `benchmark/wrappers/carl_intra_episode.py:631`, which calls `forward()` after every switch, so they are evaluated one physics substep off their training timing (measured 7.0e-6 qpos at the first switch, zero actions, w3f, seed 1; return effect on trained policies unmeasured). TrajD/DRAMA were trained on the no-forward timing (`train.py:793` → `envs/dmc_latent_factors.py:440/:451`), so each model family is evaluated on ours; only the comparators are off their training timing.

## Tests (CPU, physics level)

```bash
cd /home/jovyan/projects/DALI
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=$PWD \
  /home/jovyan/envs/dali/bin/python -m unittest discover -s drift_eval/tests -t .
```

- `test_factor_application`: the applied parameter equals the schedule exactly at reset and at
  every one of 500 steps (w3f IR-A/H-S, w2fmix H-A, quadruped phys H-S, quadruped mixed IR-S).
  Scheduled values are clamped exactly as CWM clamps them (the quadruped H-S joint_damping
  10.000000000000002 becomes 10.0). The vendored generator's sha256 is enforced.
- `test_generator_consistency`: the IR-A jump count read from the written trace equals the
  generator's, and the trace columns equal the schedule. Realised IR-S and IR-A TV agree with each
  other and with the generator to 1e-9 (w3f, quadruped_2f_phys).
- `test_worker_propagation`: the schedule reaches a spawned stepping process (w3f,
  quadruped_2f_phys, quadruped_2f_mixed). Missing payloads fail closed. A fixture shows the
  wrapper's parent global does not cross a spawn.
- `test_output_schema`:
  - raw JSON keys match a real CWM raw JSON (`.../ood_w2fmix_c2wm_final_2026-09-14/raw_ours/trajd_c2wm_final_walker_mixed_s3__lambda_0.1.json`) when present;
  - the cell schema holds;
  - CSV columns are parsed from the pinned CWM runner;
  - verification refuses a trace that differs from its schedule.
- `test_reset_seeding`: step-0 qpos equals CWM's (`suite.load` with the worker seed, and the verifier's recorded
  CWM episodes) for w3f and quadruped_2f_phys, 2 workers × 2 episodes.
- `test_row_join`: a real port output (gym-level episodes, the port's writer) joins every trace row to its
  return by (worker, ordinal) with CWM's join (mirrored verbatim from `run_eval_benchmark.py:659-688` @befabde).
- `test_parity_fixes`: no `forward` during a step; gravity table computed as CWM; quadruped
  Bernoulli draws replicate CWM's RNG call order over 400 steps; reward RNG seeded with the worker seed.
- `test_schema_ours`: condition ids, reward-mode ids, mixed-cell schedule hash, experiment/env,
  std ddof 0, envs per batch 5 and CWM's retained-episode plan.
- `test_crn_and_parity`: conditions share initial MuJoCo states per episode (w3f,
  quadruped_2f_phys). The port's Bernoulli chain equals DALI's native `make_carl_env` chain in
  wrapper types, factor configs, train ranges, RNG states, context features and selector state,
  for all five benches.

Dependencies read from CWM, with their sha256 recorded in every output:
- `config_files/latent_factor_values.yaml`
- `benchmark/wrappers/carl_intra_episode.py`
- `benchmark/wrappers/carl_reward_switch.py`
- `envs/dmc_reward_factors.py`
- `envs/schedule_channel.py`
- `benchmark/switch_recovery.py`

- `config_files/experiments/dmc.yaml`, `config_files/configure.yaml` (the generator's bench presets)

These are read from `CAUSAL_WORLD_MODEL_ROOT` (default: the pinned tree `/home/jovyan/m10_code/45e8d5d`).

**Generator (vendored).** `drift_eval/vendor/drift_generator.py` is a byte-identical copy of CWM
`benchmark/drift_generator.py` at commit befabde (befabde38bfea2e9d150fa2da41facf738c49d59; branch main after push); provenance and
sha256 are in `vendor/VENDORED.md`. It is the default. Loading refuses any file whose sha256
differs. An optional override (`DRIFT_GENERATOR_PATH` or `--generator`, e.g. the devbox worktree
`$CAUSAL_WORLD_MODEL_ROOT/benchmark/drift_generator.py` in a checkout at befabde or later) must be byte-identical or come
with its own recorded hash in `DRIFT_GENERATOR_SHA256`. The vendored copy reads its bench presets
from the CWM root; at 45e8d5d those are byte-identical to befabde's.
