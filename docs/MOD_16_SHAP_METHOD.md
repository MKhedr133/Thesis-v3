# MOD-16 — Explaining frozen difficulty regressions with SHAP

## Aim and method

The thesis predicts the assigned difficulty stage of a trial in a virtual reality supermarket. Four final multiple linear regression (MLR) equations cover Visual, Auditory, Cognitive primary and Cognitive later. These predict the *current* trial's encoded stage (D0=0, D2=1, D6=2, D10=3), not a validated optimal difficulty for a later session. MOD-16 explains existing predictions; it does not refit, tune, clip, round or replace any model.

We use `shap.LinearExplainer` with an `Independent` masker (interventional attribution convention). Its background includes every complete, eligible observation from the **26 development participants** for that model; we disable the masker's default 100-row subsampling. Visual, Auditory and Cognitive primary reference stages 0–3. Cognitive later references only stages 1–3 and includes the existing change-from-D0 predictor. Test participants never enter a background. Their trial values are used only to explain their own frozen-model predictions.

For a model with intercept b0, coefficients bj, current feature values xj and development reference means μj:

```text
reference prediction = b0 + sum(bj * μj)
SHAP contribution j  = bj * (xj - μj)
model prediction      = reference prediction + sum(SHAP contributions)
```

These signed contributions exactly decompose the additive equation under the chosen convention. The reference is the model's mean prediction over eligible development observations. It is **not** the D0 experimental baseline. We retain the same reference to explain development and unseen test observations. This approach does not redistribute attribution credit among correlated predictors; dependence-aware SHAP is deferred (Aas et al., 2021).

## Output and interpretation

Each model has two CSV files containing all development and test observations, respectively: participant and trial identifiers, encoded stage, feature values, signed contributions, reference prediction, original continuous prediction, reconstructed prediction and numerical difference. Each model also has a global bar chart comparing development and test mean absolute contributions; a development distribution plot of signed contributions; and paired local signed-contribution charts for one development and one unseen-test trial.

Local examples are selected by a rule fixed before inspecting SHAP results: the median continuous prediction in each split, with deterministic identifier tie-breaking. Mean absolute SHAP describes the *typical magnitude* of a contribution relative to the chosen reference. It does not measure unique explained variance, statistical significance, prediction accuracy or a causal effect. The test summary is descriptive because there are six independent-test participants, some contributing multiple trials.

The shared output files `development_references.csv`, `verification.csv` and `manifest.json` record the model-specific reference means, participant counts, stage eligibility, SHAP version and numerical checks. The output writer refuses to overwrite an existing nonempty directory.

## Verification

MOD-16 reads `regression_coefficients.csv`, `training_predictions.csv` and `test_predictions.csv` exported by MOD-14. MOD-13 reconstructs frozen models from the coefficient table; there is **no call to a regression-fitting function** in MOD-16.

For every development and test observation, the script verifies:
- SHAP equals the coefficient times the difference from the development reference mean.
- Reference plus contributions reconstructs the original prediction to within 1e-8 difficulty-stage points.
- The original MOD-14 prediction matches, with the same participant and model-row order.

It checks the original MOD-09 26/6 participant assignment, model-specific predictors, and allowed stage subsets, including the Cognitive later D0 exclusion. Tests use synthetic known-coefficient examples and a synthetic end-to-end MOD-14 fixture. Held-out data cannot influence background estimation or method choice.

## Running the existing pipeline

Install `requirements-dev.txt`, then run from the repository root after MOD-09 and MOD-14 have produced their original outputs:

```bash
python python/MOD_16_linear_shap.py \
  --modeling-data data/modeling_data.csv \
  --holdout-split outputs/MOD_09_participant_holdout_split/participant_holdout_split.csv \
  --mod14-output-dir outputs/MOD_14_direct_linear_regression_evaluation \
  --output-dir outputs/MOD_16_linear_shap
```

Adjust file paths to the actual research data locations. Run tests using `python -m unittest discover -s tests -p 'test_*.py'`. Research data and generated numerical results are excluded from Git; without the underlying files, no actual research feature ranking or participant explanation can be claimed.

## Scientific limitations and references

SHAP explains why a fixed equation produces its numerical estimate relative to a declared reference. Positive values raise that estimate and negative values lower it; they do not establish a behavioural intervention that would change actual difficulty. Compare the existing held-out mean absolute error (MAE), R-squared and stage-tolerance accuracy separately from explanation checks. The figures have not yet been evaluated for their usefulness to therapists.

References: Lundberg & Lee (2017), “A Unified Approach to Interpreting Model Predictions,” *NeurIPS* 30; Aas, Jullum & Løland (2021), *Artificial Intelligence* 298, 103502; Gelman (2008), *Statistics in Medicine* 27, 2865–2873; Kraha et al. (2012), *Frontiers in Psychology* 3, 44.