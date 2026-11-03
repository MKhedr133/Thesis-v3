# MOD-18 — Prospective fuzzy recommendations

## Research aim
The VR supermarket thesis investigates Visual, Auditory and Cognitive task difficulty. MOD-18 proposes the **next-session** stage after a completed trial: Increase, Decrease, Maintain or Review. A therapist retains final authority; this module does not predict treatment benefit or clinical readiness. MOD-16 SHAP and MOD-17 additive fuzzy rules explain existing **retrospective MLR predictions** and are separate from MOD-18.

## Model and participant boundaries
The four independent configurations reuse the frozen MOD-14 predictor registry: Visual (7 predictors), Auditory (4), Cognitive primary (9; D0, D2, D6, D10), Cognitive later (10; D2, D6, D10). Twenty-six development participants determine calibration and candidate comparisons; six reserved participants provide a later, locked descriptive evaluation. No held-out observation may influence rule/threshold/figure selection.

The proposed routed Cognitive workflow uses primary at D0 and later at D2/D6/D10 when an earlier Cognitive D0 baseline is available. The separate primary D2/D6/D10 figures are retained for research comparisons and are not combined numerically with Cognitive later. The historical experimental trial order may place D0 after a later stage. The code flags retrospective baseline use rather than presenting it as valid prior information.

## Candidate policy
Each model defines five two-antecedent Decrease rules and five three-antecedent Increase rules. All 40 rules are **candidate hypotheses**, including explicitly flagged exploratory rules using head turning or list consultation. Every antecedent must be a predictor in its own frozen MLR. Development-only minimum, median and maximum anchors at each assigned stage define Low/Medium/High shoulder/triangle/shoulder memberships. Min aggregates AND conditions; max pools support for Increase or Decrease. The experimental thresholds 0.10 and 0.20 are not clinical cutoffs, and membership strengths are not probabilities.

Maintain is the fallback only if the evidence is valid. Missing/non-finite inputs, out-of-development-range values, conflicting support, impossible D0/D10 transitions, unmatched high demand and questionable Cognitive later outcomes trigger Review. A Cognitive later Increase with positive performance deterioration relative to D0 is flagged for therapist review: stage-relative Low deterioration may still mean substantial loss in correctly collected products. The raw rule decision remains available beside the safeguarded outcome.

## Output files
The shared directory contains `fuzzy_calibration.csv`, `rule_registry.csv`, `cross_model_summary.csv`, `routed_recommendations.csv`, `validation_report.json` and `manifest.json`. Each of four model directories contains `recommendations.csv`, `memberships.csv`, `rule_activations.csv`, `rule_coverage.csv`, `threshold_comparison.csv`, `review_flags.csv` and `model_summary.csv`.

Three output modes:

- `--render-figure-set`: **one PNG-only folder** with eight development plots (four heatmaps, four threshold comparisons) in `development/` and 90 held-out recommendation plots in `test/<model>/`. Expected test counts: Visual 24, Auditory 24, Cognitive primary 24, Cognitive later 18. Files are `participant_<ID>_D<level>.png`.
- `--scope development --render-summary-plots`: eight aggregate figures plus numerical CSV/JSON exports, no individual development plots.
- `--scope test --render-plots`: held-out individual figures plus numerical CSV/JSON exports.

Individual figures retain concise headings, values versus stage medians, L/M/H memberships, ten rule scores and one-line decision rationale. No separate *Limitations and review* section. Plot modes enforce their scope; `--plot-threshold` controls test figures while development threshold plots compare both gates.

The compact presentation takes inspiration from Martínez et al. (2023), Martínez-Cid et al. (2024), Lee et al. (2020), Schoonderwoerd et al. (2021) and Liao et al. (2020), as considered within the 21-paper literature review. These studies support therapist-facing explanations, not our particular rule consequents or thresholds.

## Running the analysis

Once the proposed policy is documented and **frozen**, generate the single PNG folder:

```powershell
& $Python .\python\MOD_18_fuzzy_recommendations.py `
  --modeling-data $ModelingData `
  --holdout-split $HoldoutSplit `
  --output-dir ".\outputs\MOD_18_png_figures" `
  --render-figure-set `
  --enable-locked-test-evaluation `
  --frozen-policy-id "ACTUAL_FROZEN_POLICY_ID" `
  --plot-threshold 0.20
```

The CLI requires a nonempty policy identifier but does **not** verify external approval. Do not use a placeholder to bypass the independent-test safeguard. Output directory must be absent or empty.

For development-only numerical exports and eight summary figures, use `--scope development --render-summary-plots` instead, with a separate output directory.
 
## Verification and scientific limits
Synthetic tests are included for predictor restrictions, membership endpoints and ties, min/max inference, threshold endpoints, boundary cases, missing values, D0 baseline ordering, output counts, consolidated PNG-only export and plot-mode safeguards. The official private-data pipeline must be executed separately. Rule counts and stage comparisons describe behaviour, not clinical recommendation accuracy. Repeated observations from the same participant are not independent people. The 5+5 libraries and the Review policy require therapist approval before clinical use; do not change them using held-out outcomes. The MOD-18 branch must not be merged without explicit approval and verified checks.
