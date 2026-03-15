# Dynamics-Aligned Latent Imagination (DALI)

#### [[OpenReview]](https://openreview.net/forum?id=41bIzD5sit) [[arXiv]](https://arxiv.org/abs/2508.20294) [[Code]](https://github.com/frankroeder/DALI)

[Frank Röder*](https://frankroeder.github.io/), [Jan Benad](https://scholar.google.com/citations?user=FWy1Ov0AAAAJ), [Manfred Eppe](https://scholar.google.de/citations?user=fG0VWroAAAAJ), [Pradeep Kr. Banerjee*](https://scholar.google.com/citations?user=cnSjMBwAAAAJ) (&#42; equal contribution)<br/>
All the listed authors are members of the [Institute for Data Science Foundations](https://www.tuhh.de/dsf/homepage).

---

This is the official implementation of **Dynamics-Aligned Latent Imagination** (**DALI**) based on [Jax](https://github.com/jax-ml/jax).
DALI introduces a Dreamer-based architecture with a self-supervised context encoder designed to infer latent environmental
parameters, such as gravity or friction. This integration facilitates robust zero-shot adaptation to diverse and previously
unseen dynamics. Furthermore, manipulating the learned context representation enables physically plausible counterfactual imagination directly within the world model.

<p align="center">
<img src="https://github.com/user-attachments/assets/0f2ae302-4b55-40ea-98f9-c99b556b3cff">
<img src="https://github.com/user-attachments/assets/2285bb58-6943-462d-ac26-4ceeaa3ccba1">

<img src="https://github.com/user-attachments/assets/d7e2009b-4ecf-4c89-a3df-908ac58a0e0d">

</p>

### Citation
If this code or paper aids your research, please cite it as follows:
```bib
@inproceedings{
  Roder_DynamicsAlignedLatent_2025,
  title={Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization},
  author={Frank R{\"o}der and Jan Benad and Manfred Eppe and Pradeep Kr. Banerjee},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=41bIzD5sit}
}
```

## Setup
Use [uv](https://github.com/astral-sh/uv) to setup your python environment.

```bash
uv sync
uv pip install -e ./dreamerv3_compat
uv pip install -e ./
```

## Training

Run scripts in `./local_scripts/` to generate results for experts, random policies, and DALI variants.

## Record Data for Analysis

```bash
uv run -m contextual_mbrl.dreamer.record_context --logdir logs/carl_dmc_walker_double_box_enc_img_dec_img_ctxencoder_transformer_normalized/1337
```

### Log data for counterfactual dreams
```bash
uv run -m contextual_mbrl.dreamer.record_counterfactual_plausibility --logdir logs/carl_dmc_ball_in_cup_double_box_enc_img_dec_img_ctxencoder_transformer_grssm_normalized/1337 --jax.platform cpu
```

### Log data for imagined counterfactual obs trajectories
```bash
uv run -m contextual_mbrl.dreamer.record_counterfactual_plausibility_obs --logdir logs/carl_dmc_walker_double_box_enc_img_dec_img_ctxencoder_transformer_normalized/1337 --jax.platform cpu
```

### Record dataset for counterfactual obs analysis
```bash
uv run -m contextual_mbrl.dreamer.record_counterfactual_plausibility_obs_dataset --logdir logs/carl_dmc_walker_double_box_enc_img_dec_img_ctxencoder_transformer_normalized/1337 --jax.platform cpu
```

## Plots

To generate the plots, run the scripts in the `./analysis` directory.

## Acknowledgments

- cRSSM variants: Prasanna et al. 2024 - [https://github.com/sai-prasanna/dreaming_of_many_worlds](https://github.com/sai-prasanna/dreaming_of_many_worlds)
- DreamerV3: Hafner et al. 2023 - [https://github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- CARL benchmark: Benjamins et al. 2023 - [https://github.com/automl/CARL](https://github.com/automl/CARL)

---

## Benchmark Fork

> **This fork** ([`benchmark` branch](https://github.com/SSubhnil/DALI/tree/benchmark)) contains modifications for reproducing benchmark comparisons against [DRAMA](https://github.com/SSubhnil/CausalWorldModel). The original DALI code is available at [https://github.com/frankroeder/DALI](https://github.com/frankroeder/DALI).

### Modifications

1. **CARL 1.1.1 compatibility** — guarded imports for removed/renamed environments (`CARLDmcBallInCupEnv`, Box2D envs)
2. **ale-py migration** — replaced deprecated `gym.envs.atari` with `ale_py.roms` in the DreamerV3 Atari wrapper
3. **EGL rendering** — `train.py` defaults to `MUJOCO_GL=egl` and imports `dm_control` before JAX to avoid LLVM conflicts (OSMesa segfaults with JAX)
4. **Benchmark presets** — configs for Regime A (cross-episode), Regime B (intra-episode), Atari mode switching, and Procgen level switching via CausalWorldModel wrappers

### Benchmark Setup (conda)

Requires [CausalWorldModel](https://github.com/SSubhnil/CausalWorldModel) as a sibling directory (or set `CAUSAL_WORLD_MODEL_ROOT` env var).

```bash
# 1. Clone repos as siblings
git clone -b benchmark https://github.com/SSubhnil/DALI.git
git clone https://github.com/SSubhnil/CausalWorldModel.git

# 2. Create conda env
conda create -n dali python=3.10 -y
conda activate dali

# 3. Install JAX with CUDA (MUST be first)
pip install jax[cuda12]

# 4. Install benchmark requirements
pip install -r requirements_benchmark.txt

# 5. Install DALI packages
pip install -e ./dreamerv3_compat -e ./

# 6. Install EGL rendering (required for MuJoCo + JAX)
conda install -c conda-forge libegl libgl -y
```

### Running Benchmarks

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# DMC Walker — Regime A (cross-episode context switching)
python -m contextual_mbrl.dreamer.train \
  --configs carl dmc_walker benchmark_regime_a \
  --logdir logs/benchmark_regime_a_walker_s42 --seed 42

# DMC Walker — Regime B (intra-episode context switching)
python -m contextual_mbrl.dreamer.train \
  --configs carl dmc_walker benchmark_regime_b \
  --logdir logs/benchmark_regime_b_walker_s42 --seed 42

# Atari Alien — mode switching
python -m contextual_mbrl.dreamer.train \
  --configs carl benchmark_atari_alien \
  --logdir logs/benchmark_atari_alien_s42 --seed 42

# Procgen CoinRun — level switching
python -m contextual_mbrl.dreamer.train \
  --configs carl benchmark_procgen_coinrun \
  --logdir logs/benchmark_procgen_coinrun_s42 --seed 42
```
