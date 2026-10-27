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

Each model has two CSVs containing eligible development and test observations, respectively: participant and trial identifiers, encoded stage, raw feature values, signed contributions, reference prediction, original continuous prediction, reconstructed prediction and numerical difference.

Each model produces:
1. **\`contribution_distributions.png\`** — two panels, `Development (n=...)` and `Test (n=...)`, with a shared SHAP-value axis. Each dot is an eligible trial. A dot to the right of zero raises the predicted stage relative to the model-specific development reference; a dot to the left lowers it. Features are ranked using development mean absolute SHAP. The test data do not determine the ranking.
2. **\`waterfalls_test/participant_<ID>.png\`** — a cumulative waterfall for every held-out participant and model. Each trial begins at the fixed development reference; signed feature contributions lead to the original continuous prediction. The short trial heading shows the assigned difficulty (D0, D2, D6, or D10) and the fitted prediction. A single legend identifies positive and negative SHAP, the numerical reference, and the prediction line.

Plot feature labels follow the **exact Measurement column names in Section 4 of `difficulty_modelling_procedure(4).docx`**: Mental demand; Time between qualifying grabs; List recheck count; Total list recheck duration; Time to locate target; Irrelevant focus duration; Head turning; Reach duration; Reach path ratio; Relative performance change. Only features included in a given model appear on that model's plots. Raw CSV variable names and measurement units are preserved separately.

Titles and axis labels are intentionally short. Interpretation of signed contributions, dependency of repeated trials, reference selection, and scientific limitations belong in this document rather than repeated footnotes on each image.

All six held-out participants are explained for every applicable model. With the present dataset this means **24 participant-level figures and 90 trial-level waterfall panels**: D0, D2, D6 and D10 for Visual, Auditory and Cognitive primary; D2, D6 and D10 for Cognitive later. Test results are descriptive, and trials from one participant are dependent. SHAP contributions do not establish causality, predictive accuracy or therapist usefulness. Mean absolute SHAP remains calculable from the numerical attribution CSVs.

The shared output files `development_references.csv`, `verification.csv` and `manifest.json` record the model-specific reference means, participant counts, stage eligibility, SHAP version and numerical checks. The output writer refuses to overwrite an existing nonempty directory.

## Correlation diagnostics (development data only)

For each model, MOD-16 also computes **Pearson correlation coefficients** among the predictors, using the original measured values from *that model's eligible development observations*. Pearson's `r` ranges from −1 (perfect negative linear association) through 0 (no linear association) to +1 (perfect positive linear association). This descriptive calculation does not fit or modify a regression model.

Three new outputs appear inside each model directory:

- `correlations_pearson_development.csv`: the complete predictor-by-predictor correlation matrix, with original variable identifiers.
- `correlation_pairs_development.csv`: every unique predictor pair, ranked by absolute correlation, with trial and unique-participant counts.
- `correlations_pearson_development.png`: a symmetric heatmap with numeric coefficients and the exact feature names from Section 4 of `difficulty_modelling_procedure(4).docx`; the colour scale stays fixed at −1 to +1. If a feature is constant, the undefined correlation is marked `n/a`, not zero.

Test observations are excluded from these diagnostics. Each participant can contribute multiple development trials. Accordingly, the resulting trial-level correlations cannot be interpreted as correlations estimated from 103–104 independent people, and the heatmap is *not* evidence of statistical significance or causation. It is a descriptive check for possible feature dependence, which may affect how independent and conditional SHAP distribute credit. Conditional SHAP is **not** introduced here. In particular, high correlation does not alone establish that a Gaussian conditional attribution model is appropriate.

## Choice and limitations of the SHAP reference

The current reference is the model's **mean fitted prediction over complete, eligible development trials**: `intercept + sum(coefficient_j × mean_development_feature_j)`. The arithmetic mean weights eligible *trials* equally, not participants equally; Visual includes 103 observations because one development trial is missing, whereas other full-stage models include 104. The same fixed background explains development and unseen test observations.

The development mean was chosen because it is reproducible, matches the observation-level regression workflow, avoids additional assumptions about a clinically normal participant, and works for all four models. This is a defensible choice, **not a universal SHAP requirement**.

Another reference distribution, such as eligible D0 development trials or a carefully justified clinical reference population, would answer a different question and produce different contributions. For a specified reference feature vector `x_ref`, independent linear SHAP gives `reference=f(x_ref)` and `SHAP_j=coefficient_j × (x_j−x_ref_j)`. The original model prediction remains unchanged. The Cognitive later equation excludes D0, making a D0-based reference an out-of-domain choice for that model.

Crucially, **an actual trial labelled D0 (encoded stage 0), a model prediction of zero, and a SHAP reference of zero are not equivalent**. One cannot simply replace the current SHAP reference with 0 while retaining the existing contributions; this would violate additivity. A zero-reference attribution would need a defensible feature reference for which the fitted prediction is zero, or an explicitly described display-only re-centering transformation that is not falsely presented as newly computed independent SHAP.

SHAP documentation: https://shap.readthedocs.io/en/stable/generated/shap.LinearExplainer.html and https://shap.readthedocs.io/en/stable/example_notebooks/api_examples/plots/decision_plot.html. See also Aas, Jullum & Løland (2021), DOI 10.1016/j.artint.2021.103502.

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
  --output-dir outputs/MOD_16_linear_shap_v2
```

Adjust input paths as necessary and choose a fresh output directory; previous results are never overwritten. Run tests using `python -m unittest discover -s tests -p 'test_*.py'`. Research data and generated numerical results are excluded from Git; without the underlying files, no actual research feature ranking or participant explanation can be claimed.

## Scientific limitations and references

SHAP explains why a fixed equation produces its numerical estimate relative to a declared reference. Positive values raise that estimate and negative values lower it; they do not establish a behavioural intervention that would change actual difficulty. Compare the existing held-out mean absolute error (MAE), R-squared and stage-tolerance accuracy separately from explanation checks. The figures have not yet been evaluated for their usefulness to therapists.

References: Lundberg & Lee (2017), “A Unified Approach to Interpreting Model Predictions,” *NeurIPS* 30; Aas, Jullum & Løland (2021), *Artificial Intelligence* 298, 103502; Gelman (2008), *Statistics in Medicine* 27, 2865–2873; Kraha et al. (2012), *Frontiers in Psychology* 3, 44.