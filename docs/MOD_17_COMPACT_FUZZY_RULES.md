# MOD-17 — Compact additive fuzzy-rule explanations

## Aim and research context

The thesis predicts the experimentally assigned difficulty stage of a VR supermarket trial using frozen multiple linear regression (MLR) models for Visual, Auditory, Cognitive primary and Cognitive later. The stage encodings are D0=0, D2=1, D6=2 and D10=3; Cognitive later uses only stages 1–3. The participant split is frozen at 26 development and 6 independent-test participants. Outputs explain a **current fitted prediction**, not a clinically validated optimal difficulty or next-session recommendation.

MOD-17 imports the frozen MOD-13/MOD-14 registry, coefficients and measurements. It does not import MOD-16 SHAP, fit models, tune coefficients or change predictions.

## Memberships, rules and reconstruction

For each original measured predictor in each model, define exactly three categories: **Low, Medium, High**. Their anchors are the minimum, median and maximum of that model's complete, eligible development observations. These are descriptions relative to the development sample, not clinical thresholds. A category with higher raw values does not necessarily increase the predicted stage.

With the fitted equation:

    f(x) = b0 + sum_j bj * xj

The linguistic rule for feature j and category k has a numerical consequent:

    c_jk = bj * (a_jk - mj)

where a_jk is the category anchor, bj the frozen coefficient and mj the eligible development mean. Memberships w_jk(xj) interpolate linearly between adjacent anchors and sum to one. The model-specific reference is:

    r = b0 + sum_j bj * mj

The compact additive fuzzy reconstruction is:

    f_tilde(x) = r + sum_j (sum_k w_jk(xj) * c_jk)

Each rule has exactly one feature antecedent. Membership weights are compatibility values, not probabilities. With the defined interpolation, the reconstruction equals the additive linear equation to numerical precision inside calibration bounds. Tied anchors use a degenerate partition, with an inactive category if needed; a constant development feature activates Medium.

**Out-of-anchor observations:** Membership lookup clamps to the nearest development anchor; a separate, explicitly labelled linear-continuation adjustment bj * (xj - clipped_xj) is then added. This preserves the frozen MLR result without inventing a fourth linguistic category. The original measurement, flag and correction remain available in CSV outputs and the explanation. This algebraic extrapolation is not clinically validated.

The frozen models contain 7, 4, 9 and 10 predictors: **21, 12, 27 and 30 one-feature rules** respectively, **90 rules total**. This is an additive zero-order Sugeno-style representation, not an exhaustive joint Mamdani rule base.

Numerical adjustments equal coefficient-centred linear attribution under the same reference, including the independent linear SHAP contribution values used in MOD-16. Thus MOD-17 adds a *linguistic and rule-traceable representation*, not independent causal or predictive evidence. Whether it improves therapist understanding requires separate evaluation.

## Files and agreed four plots

Shared outputs:
- fuzzy_calibration.csv: original feature names, minimum/median/maximum, development means, frozen coefficients and tied-anchor flags.
- fuzzy_rules.csv: every model's rule IDs, categories, anchors and frozen numerical consequents.
- verification.csv: participant/trial counts, prediction mismatch, fuzzy reconstruction error and out-of-anchor trial counts. **No verification plot.**
- manifest.json: method, source boundaries, extrapolation and limitations.

In each model subdirectory:
- activations_development.csv / activations_test.csv: one row per trial and predictor including measured values, all three memberships, three consequents, reconstructed feature adjustment and any continuation.
- explanations_development.csv / explanations_test.csv: one row per trial, original MLR prediction, fuzzy reconstruction, reference, encoded assigned stage, errors, exceptions and short explanations.
- membership_functions.png: Low/Medium/High curves and development anchors for **every predictor**.
- rule_activations_test/participant_<ID>.png: **all six test participants**, each with all eligible trials and predictor-level memberships.
- fuzzy_explanations_test/participant_<ID>.png: **all six test participants**, each with trial measurements, active rules, signed adjustments, reference and predicted stages.
- rule_usage_development_vs_test.png: development/test heatmaps side by side, shared 0–1 colour scale, one mean membership per participant before aggregation, no pooling for threshold definition.

The proposed outputs include 24 test-participant activation figures, 24 test-participant explanation figures, 4 membership atlases and 4 paired usage heatmaps. Cognitive later excludes D0. We do not duplicate SHAP waterfalls or feature-importance rankings.

## Execution

Use the original MOD-14 modelling data, holdout split and three untouched MOD-14 CSV outputs. Example command:

    python python/MOD_17_compact_fuzzy_rules.py \
      --modeling-data modeling_data.csv \
      --holdout-split participant_holdout_split.csv \
      --mod14-output-dir outputs/MOD_14_direct_linear_regression_evaluation \
      --output-dir outputs/MOD_17_compact_fuzzy_rules

The source MOD-14 directory must contain regression_coefficients.csv, training_predictions.csv and test_predictions.csv. The program refuses to overwrite a nonempty output directory. It validates the frozen split, model registry, eligible stages and original prediction order/numbers before saving any outputs.

## Scientific and evaluation limits

The 26 development participants alone establish category anchors; the six independent-test participants never calibrate thresholds or drive design choices. Held-out heatmaps are descriptive. Repeated trials from one participant are not independent people; heatmaps average within participants first. An exact fuzzy reconstruction does not establish that the model was accurate at predicting assigned stages, that the model is causal, or that the explanation is understandable. Unobserved values beyond development anchors should be interpreted cautiously, even when algebraically reconstructed.

The associated studies by Martínez et al. (2023) and Martínez-Cid et al. (2024) demonstrate fuzzy rehabilitation recommendation systems with rule-based explanations. Their policies and clinical thresholds cannot be directly transferred here: this thesis uses linguistic rules to represent an existing frozen MLR prediction. The corresponding Google Drive critical readings discuss rule complexity, explanation completeness and therapist comprehension.
