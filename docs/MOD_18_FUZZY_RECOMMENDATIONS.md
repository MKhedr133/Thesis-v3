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

The plotting modes are separate:
- **Development:** `--render-summary-plots` creates **20 aggregate figures** (five per model) in `<model>/summary_figures/`: `rule_activation_heatmap.png`, `threshold_comparison.png`, `recommendation_distribution.png`, `review_flag_summary.png`, `action_support_scatter.png`. There are **no individual development plots**.
- **Locked test:** `--render-plots` creates one compact plot per participant and assigned stage under `<model>/figures/D<level>/participant_<ID>_D<level>.png`, with **90 expected figures**: 24 Visual, 24 Auditory, 24 Cognitive primary and 18 Cognitive later. Each shows observed values and medians, L/M/H memberships, ten rule scores, action supports and a brief reason or Review status, without redundant text or a standalone limitations section.

Plot modes reject the wrong scope and cannot be combined. Omitting both flags produces numerical exports only. `--plot-threshold` controls the displayed individual decision and selected aggregate charts; both thresholds remain in the CSV analyses.

The compact presentation takes inspiration from Martínez et al. (2023), Martínez-Cid et al. (2024), Lee et al. (2020), Schoonderwoerd et al. (2021) and Liao et al. (2020), as considered within the 21-paper literature review. These studies support therapist-facing explanations, not our particular rule consequents or thresholds.

## Running the analysis
Development-only exports:

```bash
python python/MOD_18_fuzzy_recommendations.py \
  --modeling-data data/modeling_data.csv \
  --holdout-split data/participant_holdout_split.csv \
  --output-dir outputs/MOD_18_development \
  --scope development --render-summary-plots --plot-threshold 0.20
```

**Do not run held-out evaluation before policy approval and freeze.** After a recorded freeze, it can generate individual test-stage figures:

```bash
python python/MOD_18_fuzzy_recommendations.py \
  --modeling-data data/modeling_data.csv \
  --holdout-split data/participant_holdout_split.csv \
  --output-dir outputs/MOD_18_locked_test \
  --scope test --enable-locked-test-evaluation \
  --frozen-policy-id approved-version \
  --render-plots --plot-threshold 0.20
```

Replace `approved-version` with an actually approved and documented identifier. Outputs cannot overwrite a nonempty directory.

## Verification and scientific limits
Synthetic tests are included for predictor restrictions, membership endpoints and ties, min/max inference, threshold endpoints, boundary cases, missing values, D0 baseline ordering, output counts, summary-plot generation and plot-mode guards. The official private-data pipeline must be executed separately. Rule counts and stage comparisons describe behaviour, not clinical recommendation accuracy. Repeated observations from the same participant are not independent people. The 5+5 libraries and the Review policy require therapist approval before clinical use; do not change them using held-out outcomes. The MOD-18 branch must not be merged without explicit approval and verified checks.
