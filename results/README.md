# Results

This directory contains public baseline outputs for ThousandWorlds.

- `models/<subset>/<method>/`: the config each run used, its prediction archive, and its metrics JSON.
- `tables/`: paper-oriented metric tables grouped by subset, protocol, and metric.
- `scores.csv`: flat per-variable baseline scores generated from the metrics JSON files.
- `scores_5seeds.csv`: the same score format with a `seed` column. Learned
  baselines include seeds 0-4 here; these are the scores used for the paper
  results. Only the seed-0 outputs are distributed, so the `*_path` columns
  are populated for seed 0 and empty for seeds 1-4.

The checked-in configs, predictions, metrics JSON files, and rendered tables are
the seed-0 outputs. Use `scores_5seeds.csv` for seed-averaged numbers matching the paper.

Released configurations are listed in `thousandworlds/models/README.md`.

`models/*/*/predictions.npz` follows the standard ThousandWorlds submission
format: `predictions`, `simulation_id`, and `field_names`.
