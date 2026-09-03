# Model Details

Short descriptions of the released baselines. The config and metrics of each
released run are under `results/models/<subset>/<method>/`; the entry point is
`thousandworlds/run_model.py`.

All models use the benchmark transforms from `thousandworlds/preprocessing.py`
and score through `thousandworlds/evaluate.py`.

## Training Mean

`train_mean` predicts the same per-field training mean for every test case. See
`train_mean.py`.

## kNN

`knn` averages nearby training examples after standardizing the input
parameters. The public CV sweep uses `k = 1, 2, 3, 5, 10` and
`gcm_penalty = 0.0, 0.3, 1.0, 3.0, 10.0`; nonzero penalty values make same-GCM
neighbors closer in the multi-GCM subsets. See `knn.py`.

## PCA-Ridge

`pca_ridge` compresses T21 spectral coefficients with PPCA, then predicts the
latent scores with ridge regression. See `pca_ridge.py` and `_ppca.py`.

## PCA-MLP

`pca_mlp` uses the same PPCA representation as PCA-Ridge, but maps inputs to
latent scores with a two-hidden-layer MLP. See `pca_mlp.py`.

## PCA-GBT

`pca_gbt` uses the same PPCA representation as PCA-Ridge and PCA-MLP, but maps
inputs to latent scores with gradient-boosted regression trees (one
`HistGradientBoostingRegressor` per latent component, fit in parallel). It is
motivated by the *regime transitions* in the dataset (temperate / snowball /
runaway): axis-aligned trees can place splits at the transition thresholds that
smooth regressors blur. `learning_rate` and `max_leaf_nodes` are tuned per
subset by a 5-fold CV sweep (objective `equal_group_normalized_rmse`, the same
convention as PCA-Ridge / kNN); the chosen values, grids, and fold scores are
written to `config.json` (`CV_sweep` + `best`). See `pca_gbt.py`.

## Coord-MLP

`coord_mlp` predicts one grid value at a time from planet inputs, GCM identity,
field identity, level, latitude, and longitude. See `coord_mlp.py`.

## Coord-DeepONet

`coord_deeponet` is the learned-basis version of Coord-MLP: a branch network
encodes the planet/GCM input and a trunk network encodes the field/grid query.
See `coord_deeponet.py`.

## PPCA-ICM

`ppca_icm` predicts PPCA latent scores with a Gaussian process using a
Matérn-5/2 input kernel and a GCM coregionalization term. The released configs
use 64 posterior samples. See `ppca_icm.py`.

For deterministic metrics, `evaluate.py` uses `predictions_mean.npz` when a
probabilistic method provides one; probabilistic metrics use the ensemble in
`predictions.npz`.

## GPLFR

`gplfr` fits GPLFR by MAP in latent space, with fixed variable-group weights
and no output coregionalization term, and writes posterior samples in the
standard prediction format. See `gplfr.py` and `_gplfr_core.py`.

The released configurations are Multi-partial (Matérn-5/2, 115 steps,
β = 0.10, latent nugget = 0.10), Multi-complete (Matérn-5/2, 120 steps,
β = 0.10, latent nugget = 0.03), and Single-complete (Matérn-3/2, 30 steps,
β = 0.03, latent nugget = 0.03).

## Conv-Decoder

`conv_decoder` maps the planet and GCM inputs to the latitude--longitude
fields with a learned upsampling convolutional decoder. See `conv_decoder.py`.

## SFNO

`sfno` applies spherical Fourier neural operator blocks on the native  
Legendre--Gauss grid, with a learned latitude--longitude embedding. See  
`sfno.py`.

