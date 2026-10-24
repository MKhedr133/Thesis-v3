# MOD-14 accuracy sensitivity: three tolerances

## Research purpose

This extension adds an interpretable percentage metric to the final multiple linear regression (MLR) results for the virtual-reality supermarket thesis. It reports how often the original continuous prediction lies close to the observed difficulty stage. The three conditions are Visual, Auditory and Cognitive. The final MOD-14 evaluation has **four** specifications: Visual, Auditory, Cognitive initial and Cognitive later. Cognitive initial uses D0/D2/D6/D10 (stages 0–3). Cognitive later uses D2/D6/D10 (stages 1–3) and a performance-change predictor requiring a D0 baseline. Do not merge their results or compare them as though they evaluate identical trials.

The existing models were fitted on **26 development participants** and tested on **six independent participants**. Their training results are direct fitted-training estimates, not bootstrap or out-of-bag (OOB) estimates. This analysis does not fit models, modify predictors, recalculate coefficients, or alter the participant split.

## Metric definition

For each eligible trial, let `y` be its observed stage, `y_hat` the original continuous MLR prediction and `t` the error tolerance:

```text
accuracy_within_t_percent = 100 * count(abs(y - y_hat) < t) / N
```

The three **predefined tolerances are 0.25, 0.50 and 0.75 difficulty stages**. The inequality is strict: a prediction exactly `t` stages away does not count as correct. There is **no rounding or clipping** of predictions. Each eligible trial counts once. Missing, nonfinite or inconsistent values cause an error instead of silently reducing the denominator.

A higher tolerance gives an equal or higher percentage even though the models do not improve. The 0.50 threshold has an intuitive interpretation for equally spaced stage codes; 0.25 and 0.75 test stricter and looser definitions. These are descriptive thresholds, not model hyperparameters. Do not choose the threshold by maximizing performance on the six independent-test participants.

The metrics use different weighting. New accuracy is pooled across trials; existing mean absolute error (MAE) is participant-balanced, with each participant contributing equally; existing R² is pooled across trials. MAE and R² continue to use continuous predictions, unchanged.

## Frozen inputs and safeguards

The script reads `training_predictions.csv`, `test_predictions.csv` and `train_test_performance.csv` produced by MOD-14. It imports the existing MOD-14 model registry and metric functions. It checks model and condition names, model-specific valid stages, finite predictions, saved errors, duplicate participant/model/stage keys, consistent participant membership across model results, disjoint 26/6 participant groups, expected row counts and out-of-range counts. It independently recomputes MAE/R² from saved predictions and compares them against the frozen performance table.

These checks establish consistency **within the supplied saved records**. They cannot independently verify participant membership against the original MOD-09 split file or a separate trial-to-prediction join without those external records. No test observations are used for fitting or model selection.

## Windows PowerShell commands

Run from the existing local checkout; no ZIP or manual source-file download is needed:

```powershell
Set-Location "D:\TU\MScRobotics\Thesis_files\Rebuild_v3\Thesis"
git fetch origin
git switch --track origin/feature/mod14-accuracy-sensitivity

python -m unittest discover -s tests -p "test_mod14_accuracy_sensitivity.py" -v

python python/MOD_14_accuracy_sensitivity.py `
  --mod14-output-dir "outputs/MOD_14_direct_linear_regression_evaluation" `
  --output-dir "outputs/MOD_14_accuracy_sensitivity"
```

If the feature branch already exists locally, use `git switch feature/mod14-accuracy-sensitivity` and `git pull --ff-only`. The script expects original unsuffixed MOD-14 CSV filenames in the specified input directory. To regenerate the *extension's own* results, add `--overwrite`. Never point the new `--output-dir` at the frozen MOD-14 directory.

## Outputs

All generated files live in a separate `outputs/MOD_14_accuracy_sensitivity/` directory (ignored by Git):

- `performance_with_accuracy.csv`: original MAE/R², row and participant counts, and three percentages for each split and model
- `accuracy_by_threshold.csv`: 24 model/split/threshold combinations, each with correct counts, denominators and percentages
- `accuracy_trial_audit.csv`: per-trial flags; **contains participant IDs and must remain private**
- `analysis_manifest.json`: metric definitions, weighting and source CSV SHA-256 hashes
- `accuracy_lt_0p25.png/.svg`, `accuracy_lt_0p50.png/.svg`, `accuracy_lt_0p75.png/.svg`: three grouped development/test accuracy charts
- `accuracy_sensitivity_overview.png/.svg`: two-panel development/test threshold comparison
- `training_vs_test_mae.png/.svg`: participant-balanced MAE chart
- `training_vs_test_r2.png/.svg`: pooled R² chart

There are **16 output files**, including six high-resolution figures in PNG and SVG formats. Original MOD-14 outputs remain unchanged.

## Interpretation limits

The independent test covers six participants with repeated trials. Apparent score differences do not establish population-level differences or overfitting. Cognitive later excludes baseline trials, so its results are not a like-for-like comparison with Cognitive initial. Correctly predicting the **observed** difficulty stage does not validate a system for recommending the **next** difficulty. Present all three tolerances to Beatrice and agree on their substantive meaning before choosing a primary reporting convention.