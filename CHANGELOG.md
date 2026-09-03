# Changelog

## v2.0.0 (2026-09-03)

v2.0.0 replaces v1.0.0. Changes relative to v1.0.0:

- Corrected bug in ExoCAM cloud fractions, which had been incorrectly treated as percentages rather than fractions, leaving a factor of 100 missing.
- Found a bug in the Mak et al. (2024) UM simulations; most simulations regenerated and
  a small number removed.
- Applied a tighter convergence criterion (|ASR − OLR| < 20 W m^-2), removing
  some simulations, mostly ExoPlaSim.
- Added a small number of simulations at higher surface gravity.
- Added the ConvDec and SFNO baselines.
- Retuned all baselines on every subset and ablation, as documented in the
  paper.
