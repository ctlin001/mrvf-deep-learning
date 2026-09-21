# How the refactor was checked against the original notebooks

The library in `src/mrvf/` was extracted from the notebooks `final_TripleRegime_ABC_Train`,
`final_TripleRegime_NoiseFree_Train` and `final_ablation_rmse_vs_snr` (now in `notebooks/legacy/superseded/`).
To check that nothing changed numerically, the **original notebook cells were executed unchanged** and their outputs
compared with the refactored code on the real simulation data.

"Identical" below means `numpy.array_equal` / `torch.equal`: bit-for-bit, not "close".

## What was compared

| # | Check | Data | Result |
|---|---|---|---|
| 1 | Echo-timing split (T_A, T_B, T_C_rel) | `echotimes.mat` | identical |
| 2 | `euclidean_norm`, `params_scale`, `params_inverse`, `filter_param_range`, `clean_data` | adversarial random data incl. NaN, inf, out-of-range | identical (values and dtypes) |
| 3 | R2\*_A / B / C features | 200,000 real noisy signals (SNR 50) | identical, max difference 0.0 |
| 4 | Scaled features and the 43-value input | same 200,000 signals | identical |
| 5 | `TripleRegimeModel` | same seed → same initial weights; shipped **noisy** and **noise-free** checkpoints | identical layer names, identical outputs |
| 6 | `Conv1DModel` (DL-4p) | shipped DL-4p checkpoint | identical outputs |
| 7 | Batched inference | 50,000 rows | identical |
| 8 | Dictionary matching | 1,499,264-entry dictionary, 3,000 queries | identical matches |
| 9 | **Mixed-SNR training set** (SNR 20 & 150, seeded 3 M → 1.6 M subsample, split) | real dictionaries | all 7 arrays identical |
| 10 | **Noise-free training set** (whole dictionary, split) | real dictionary | all 7 arrays identical |
| 11 | **Training loop**, 5 epochs, seeded, CPU | 30,000 real rows | identical loss history and identical best weights |
| 12 | **RMSE vs SNR**, 4 methods × 4 SNR levels | full run of `scripts/run_pipeline.py evaluate` on the shipped checkpoints vs the saved `results/ablation_results/ablation_rmse.json` | 64 values, **max difference 0** |
| 13 | Per-SNR RMSE over every dictionary entry (`per_snr_rmse.json`) | shipped noisy and noise-free checkpoints vs their saved JSON files | 32 values, max difference 0 |
| 14 | Feature ablation (8 conditions × 4 parameters) | 60,000 real rows | identical labels, identical values |

Also run:

- **Unit tests**: `python -m unittest discover -s tests` (16 tests; no data needed).
- **CLI end to end**: `run_pipeline.py all` at small scale, training both variants and evaluating them; every expected
  file was written (models, JSON metrics, figures).
- **Both notebooks executed** with `nbconvert`: notebook 02 in full (including its built-in assertion against the
  published numbers), notebook 01 in `QUICK` mode.

## GPU training is not bit-identical (and cannot be)

On CPU the training loop is bit-identical to the original. On GPU, cuDNN kernels are not deterministic, so two runs
of the *same* code with the *same* seed differ slightly. Measured over 5 epochs on 30,000 rows (largest difference in
validation loss between two runs):

| Comparison | max \|val-loss difference\| |
|---|---|
| original code vs original code (two GPU runs) | 2.2 × 10⁻² |
| refactored code vs refactored code (two GPU runs) | 1.0 × 10⁻² |
| original code vs refactored code (all four pairings) | 1.9 × 10⁻² |

The original-vs-refactored gap is inside the run-to-run noise of the *original* code alone, so there is no detectable
difference on GPU either.

## What was *not* checked

- **A full 200-epoch retrain.** It takes up to ~2 h per model, and it could not match the shipped weights anyway: the
  original notebooks did not seed the dataset subsampling, network initialisation or batch order. Equivalence rests on
  checks 9–11 (same data in, same training procedure) and 12–13 (same trained weights in, same numbers out).
- **Figures.** The library draws its own figures; they are not pixel-copies of the original ones.
- **The legacy notebooks** (`notebooks/legacy/`) are unchanged apart from updated paths and a working-directory cell. They were
  not re-run, since the in vivo ones need human-subject data that is not in this repository.

## Deliberate differences from the original code

1. **Seeds.** Dataset subsampling, network initialisation and batch shuffling are seeded (default 42).
2. **Row alignment.** The original mixed-SNR dataset builder dropped non-finite rows from the signal array and then cut the
   feature arrays with `[:n]`, which would misalign them if any row had been dropped. The library masks all arrays together. No
   row is dropped on the real dictionaries, so the outputs are identical (check 9).
3. **One shared row selection per SNR level.** All methods are scored on exactly the same rows. The original computed the
   selection separately for DM/DL-4p and for the triple-regime models, which gives the same rows when nothing is non-finite (true here).
4. **GPU memory is released after training** (`training.release_gpu_memory`). Without this, running `train` and then
   `evaluate` in one process made dictionary matching ~20× slower on a 6 GB card.
5. **Paired significance tests use the errors as computed.** In `superseded/final_ablation_rmse_vs_snr`, cell 25 multiplies
   DM's T2 errors by 1.5 (the comment says 2×) before the Wilcoxon tests and the MAE bar chart. `evaluation.paired_error_tests`
   does not rescale anything, so its output differs from that notebook's for the DM-vs-Triple T2 comparison; the RMSE results
   (check 12) are unaffected, because they are computed before that cell.
