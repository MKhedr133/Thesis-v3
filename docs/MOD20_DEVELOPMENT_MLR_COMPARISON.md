# MOD-20 Development MLR feature comparison

This analysis compares three multiple linear regression (MLR) approaches for predicting assigned difficulty stage (0–3). It uses trial features from the FE-04 all-grab export and evaluates the Visual, Auditory and Cognitive conditions separately. It leaves the frozen MOD-14 models unchanged and does not use age group or TMT-B.

## Comparison

In each of five participant-grouped cross-validation rounds, the full MLR uses all ten trial features. The screened MLR uses features with fixed-effect p-values below 0.05 in a mixed model fitted to the training participants with a participant random intercept. The add-back MLR restores screened-out members of one predefined pair and refits the equation.

| Condition | Add-back pair |
|---|---|
| Visual | Product search duration and head rotation during search |
| Auditory | Time between grabs and product search duration |
| Cognitive | Shopping list recheck count and total recheck time |

All candidates in a round train and score on the same complete cases. The output counts trials with all features needed by each candidate. It lists that count separately from the trials used to compare prediction errors. The mixed-model p-values are screening statistics, not evidence that a predictor is useless.

The selected approach must have fewer predictors on average than the full MLR and a participant-balanced MAE no more than 0.05 stage units higher. If both alternatives qualify, prefer fewer predictors, then lower MAE. If they tie on both, prefer the screened equation. If neither qualifies, keep the full equation. Reapply the selected approach to the eligible trials of all 26 development participants to estimate the final coefficients. The six reserved participants do not enter fitting, scoring or selection. The cross-validation results are development comparisons, not independent test performance.

## Run locally

From the repository root, provide the FE-04 export and its matching manifest. The comparison refuses an older export or a CSV that does not match its manifest. Supply the frozen MOD-09 participant split and a new output directory.

```powershell
python python/MOD_20_development_mlr_feature_comparison.py `
    --trial-features path/to/trial_features.csv `
    --feature-manifest path/to/feature_extraction_manifest.json `
    --participant-split outputs/mod09_participant_holdout/participant_holdout_split.csv `
    --output-dir outputs/mod20_development_mlr_features
```

Start with `comparison_summary.csv`. The other outputs record each round's predictor set and coverage, development predictions, mixed-model p-values and fit warnings, final coefficients, and input hashes. If the screen fails to converge or produces invalid p-values, the analysis stops rather than treating the affected features as non-significant. Keep the input data and generated CSVs local.
