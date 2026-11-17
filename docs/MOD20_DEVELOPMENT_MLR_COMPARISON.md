# MOD-20 Development MLR feature comparison

This analysis compares three multiple linear regression (MLR) approaches for predicting assigned difficulty stage (0–3). It uses trial features from the FE-04 all-grab export and evaluates the Visual, Auditory and Cognitive conditions separately. It leaves the frozen MOD-14 models unchanged and does not use age group or TMT-B.

## Comparison

In each of five participant-grouped cross-validation rounds, the full MLR uses all ten trial features. The screened MLR uses features with fixed-effect p-values below 0.05 in a mixed model fitted to the training participants with a participant random intercept. The add-back MLR restores screened-out members of one predefined pair and refits the equation.

| Condition | Add-back pair |
|---|---|
| Visual | Product search duration and head rotation during search |
| Auditory | Time between grabs and product search duration |
| Cognitive | Shopping list recheck count and total recheck time |

All candidates in a round train and score on the same complete cases. The output counts trials with all features needed by each candidate. It lists that count separately from the trials used to compare prediction errors. The p-values guide feature screening. They do not establish whether keeping a predictor improves predictions.

An initial development run returned an invalid p-value in Visual round 2 despite convergence. The estimated participant variance was close to zero. MOD-20 now uses a fallback only when a converged mixed model returns at least one non-finite predictor p-value and the random intercept accounts for at most `1e-8` of the total estimated variance (participant plus residual). This is a numerical tolerance, not a test for whether participants differ. In that case, MOD-20 fits OLS on the same training trials and recalculates p-values for all ten predictors using standard errors clustered by participant and a small-sample correction. The screening cutoff stays at 0.05. The fallback can also apply when screening the final development refit. The screen audit records the method used, the original fit diagnostics and which original p-values were finite.

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

Start with `comparison_summary.csv`. To see where the fallback ran, look for `ols_participant_clustered` in the `screening_method` column of `mixed_model_screen.csv`. The other outputs record each round's predictor set and coverage, development predictions, final coefficients, and input hashes. The analysis stops if the mixed model fails to converge, the p-values are invalid outside the numerical boundary, or the fallback also returns invalid p-values. Keep the input data and generated CSVs local.

## Report outputs

The same command writes a vector PDF figure and a LaTeX table to the output directory. Both use the values from `comparison_summary.csv`.

- `comparison_mae_change.pdf` shows how participant-balanced MAE changed relative to the full MLR. Each point summarises one candidate's results across the development cross-validation rounds. The dashed line marks the allowed increase of +0.05 difficulty-stage units. This margin guides model selection and does not establish statistical equivalence.
- `comparison_table.tex` gives the absolute MAE, mean predictor count and shared scoring coverage for each equation. The selected equation is in bold. Include the table with `\input{path/to/comparison_table.tex}` and load `booktabs` in the LaTeX preamble.

When including the figure, state in the caption that the comparison used participant-grouped cross-validation on the 26 development participants. Within each condition, all candidates were scored on the same trials. The six reserved participants were not evaluated.
