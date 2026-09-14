# Vendored: drift_generator.py

- file: drift_generator.py
- source_repo: CausalWorldModel (github.com/SSubhnil/CausalWorldModel)
- source_path: benchmark/drift_generator.py
- commit: befabde38bfea2e9d150fa2da41facf738c49d59 (branch main after push)
- sha256: 39b72cea67126f781605a2ebf7c658fa316ed380e9e2ee3badec132c561e09f6
- vendored: 2026-09-14
- note: byte-identical copy; never edit. `drift_eval/protocol.py` refuses to load it if the sha256
  differs. Its presets (`config_files/experiments/dmc.yaml`, `configure.yaml`,
  `latent_factor_values.yaml`) are read from `CAUSAL_WORLD_MODEL_ROOT`; at 45e8d5d they are
  byte-identical to befabde's.
