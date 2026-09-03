# ThousandWorlds

Benchmark for exoplanet climate emulation.

## Install

```bash
pip install -e .
```

For model reproduction:

```bash
pip install -e '.[models]'
python -m thousandworlds.run_model train_mean single-complete
```

For notebooks only:

```bash
pip install -e '.[notebooks]'
```

## Dataset download

The benchmark dataset lives on Hugging Face:

- Hugging Face DOI: `https://doi.org/10.57967/hf/8695`

`tw.download_dataset(...)` / `tw.download_baselines(...)` fetch the pinned archives from the Hub via `huggingface_hub` (cached under `~/.cache/huggingface`, integrity-checked). The repo/revision are pinned in `data.py` (`HF_REPO_ID`, `HF_REVISION`).

Runtime dependencies are intentionally minimal: `numpy`, `pandas`, and `huggingface_hub`.

## Hugging Face dataset card

The Hub dataset card is version-controlled in `HF_CARD.md` (repo root): edit it there and re-upload it as `README.md` to `es833/ThousandWorlds`. Don't hand-edit the card on the Hub, or this copy goes stale. The card's `configs:` block points the dataset viewer at a root-level `inputs.csv` (the `input_planets` config, one `all` split), so upload `inputs.csv` alongside the card (one `HfApi().create_commit` with both ops).

## Baseline notes

- PPCA-ICM results use the raw posterior spread; no post-hoc calibration is
  applied, and the released configs contain no calibration settings.

## Quickstart

```python
from pathlib import Path
import numpy as np
import thousandworlds as tw

bundle = tw.load(
    subset="single-complete",
    protocol="standard",
    data_dir=Path("dataset"),
)

pred = np.broadcast_to(bundle.Y_train.mean(axis=0), bundle.Y_test.shape)
scores = tw.evaluate.rmse(pred, bundle.Y_test, bundle.field_mask_test, bundle.field_names)
scores['per_variable']
```

Metrics are exposed per field and per variable, where a variable is a physical quantity and 3D variables are averaged over pressure levels.

`data_dir=Path("dataset")` resolves both from the `thousandworlds/` source-tree root and from a parent directory after `pip install -e thousandworlds`.

See `notebooks/pca_mlp.ipynb` for a model-training example that actually fits the public PCA-MLP model via `python -m thousandworlds.run_model`.

## Structure

```
thousandworlds/       # Python package
  data.py             # download + load dataset
  preprocessing.py    # input/output transforms, normalisation
  spectral.py         # spectral coefficients <-> gridded fields
  evaluate.py         # RMSE, energy score, relative energy score
  run_model.py        # python -m thousandworlds.run_model
  models/             # model implementations
  assets/             # precomputed SHT matrix, latitude weights

dataset/              # benchmark splits, norm stats, spectral assets
results/              # pre-computed predictions per method
notebooks/            # quickstart.ipynb, pca_mlp.ipynb, explore_trappist1e.ipynb
tests/                # test suite
```
