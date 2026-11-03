# MOD-18 — Prospective fuzzy recommendations

## Research aim
The VR supermarket thesis investigates Visual, Auditory and Cognitive task difficulty. MOD-18 proposes one of **three next-session actions** after a completed trial: Increase, Decrease or Maintain. **Review is reserved for impossible transitions** (a Stage 0 decrease or Stage 3 increase). **Unavailable** is a separate data-quality status when an action cannot be calculated. Neither is a fourth action. A therapist retains final authority; this module does not predict treatment benefit or clinical readiness. MOD-16 SHAP and MOD-17 additive fuzzy rules explain existing **retrospective MLR predictions** and are separate from MOD-18.

## Model and participant boundaries
The four independent configurations reuse the frozen MOD-14 predictor registry: Visual (7 predictors), Auditory (4), Cognitive primary (9 predictors) and Cognitive later (10 predictors). For **MOD-18 recommendation routing**, primary is evaluated only at Stage 0 (D0), while later is evaluated at Stages 1–3 (D2, D6, D10). The full frozen MOD-14 modelling registry is unchanged. Twenty-six development participants determine calibration and candidate comparisons; six reserved participants provide a later, locked descriptive evaluation. No held-out observation may influence rule/threshold/figure selection.

The Cognitive workflow uses primary at D0 and later at D2/D6/D10. The Cognitive later predictor `performance_change_from_d0_percentage_points` is already computed relative to the **same participant's Cognitive D0 performance** in the modelling dataset. MOD-18 confirms there is exactly one same-participant, same-condition D0 record; it does not require `trial_order` or replace a missing baseline with a different participant's measurement. The historical Stage 0 trial may have occurred after a later-stage trial. Such findings are retrospective baseline comparisons and cannot establish prospective availability. MOD-18 does not alter the frozen MOD-14 model registry.

## Candidate policy
Each model defines five two-antecedent Decrease rules and five three-antecedent Increase rules. All 40 rules are **candidate hypotheses**, including explicitly flagged exploratory rules using head turning or list consultation. Every antecedent must be a predictor in its own frozen MLR. Development-only minimum, median and maximum anchors at each assigned stage define Low/Medium/High shoulder/triangle/shoulder memberships. Min aggregates AND conditions; max pools support for Increase or Decrease. The experimental thresholds 0.10 and 0.20 are not clinical cutoffs, and membership strengths are not probabilities.

**Three-action decision:** the model compares the maximum Increase and Decrease rule strengths against thresholds 0.10 and 0.20. If neither reaches the threshold, it returns Maintain. If both exceed it, the stronger support wins; an exact tie results in Maintain and is documented in `diagnostic_flags`. **Review** occurs only if the chosen action is Decrease at Stage 0 or Increase at Stage 3.

Missing/non-finite predictors or rule support, or a missing/duplicate participant-specific D0 reference for Cognitive later, produce **Unavailable**, with no proposed stage. This indicates that an action could not be evaluated; it is not a recommendation or a Review decision. Other warnings—out-of-development-range inputs, competing rule support, unmatched high demand or an Increase despite deteriorated performance—are recorded in `diagnostic_flags` but do not automatically override a computable action. `review_flags` is reserved for `LOWER_BOUNDARY` or `UPPER_BOUNDARY`.

This policy revision is version `candidate-v0.2-boundary-review`. It deliberately issues more recommendations than the earlier safety-oriented proposal. These actions remain candidate hypotheses, not clinically validated therapy guidance. The independent test participants may be used only under an approved policy-freeze process and must not determine subsequent rule or threshold changes.

## Output files
The shared directory contains `fuzzy_calibration.csv`, `rule_registry.csv`, `cross_model_summary.csv`, `routed_recommendations.csv`, `validation_report.json` and `manifest.json`. Each of four model directories contains `recommendations.csv`, `memberships.csv`, `rule_activations.csv`, `rule_coverage.csv`, `threshold_comparison.csv`, `review_flags.csv`, `unavailable_cases.csv` and `model_summary.csv`.

Three output modes:

- `--render-figure-set`: **one output folder with 80 PNG plots and auditable CSV/JSON data**. The `summary/` directory contains four **recommendation distribution** figures (both cohorts) and four development-only **rule-activation heatmaps**. Each distribution PNG has two side-by-side panels: development (26 participants) and held-out test (six), with grouped—not stacked—bars for thresholds 0.10 and 0.20. The categories are Increase, Maintain and Decrease; Review and Unavailable are not plotted as adjustment actions. Each bar shows count and percentage. Normally these are **safeguarded actions** with the denominator restricted to non-Review participant-stage observations at that threshold. **If all observations require boundary Review**, the chart instead shows hatched **provisional raw actions** and the Review count; these are not approved recommendations. Unavailable cases have separate counts and are never converted into provisional recommendations. Where no action can be computed, counts are zero and percentages undefined. A participant may contribute more than one stage; counts are not numbers of distinct participants. Both panels are descriptive; **never choose or revise the threshold based on test results**.
- Each heatmap uses one panel per eligible stage, ten rule rows and one column per **development observation** (26 columns per stage). Every cell displays its activation strength to two decimal places; missing values are marked with a dash. Rule names are explicit, for example **Decrease rule 1 (D1)** and **Increase rule 1 (I1)**. The internal D1–D5/I1–I5 identifiers remain stable. Heatmaps are not stage-mean aggregates.
- The `test/<model>/` directories contain 72 compact individual PNGs: Visual 24; Auditory 24; Cognitive primary 6 (Stage 0 only); Cognitive later 18 (Stages 1–3 only). Each card shows (1) assigned stage, (2) safeguarded action and next stage when applicable, (3) strongest **written IF–THEN rule** with its strength and (4) activation strengths of the *other four rules of the same action*. The card and the rule registry also spell out the action associated with each rule identifier. Maintain has no Maintain rule, so its card explains the threshold fallback and shows the strongest non-qualifying rules. Review indicates an impossible boundary transition and may show provisional rule evidence. Unavailable identifies missing or invalid data and makes no stage proposal.
- `--scope development --render-summary-plots`: development-only heatmaps and threshold-comparison bar charts plus CSV/JSON exports; no test data or individual development figures. `--scope test --render-plots`: individual test cards plus CSV/JSON exports, guarded by policy freeze.

Figures avoid explanatory paragraphs, repeated provenance and separate limitations blocks. The combined export requires a previously documented policy freeze. It writes figures to `summary/` (eight PNGs) and `test/<model>/` (72 PNGs), and numerical files to `data/`:

- `data/recommendation_distribution.csv` records model/scope/threshold/action counts, denominators, percentages, `basis` (`safeguarded` or `provisional`), `review_count`, `unavailable_count` and `observation_count`. Review remains excluded from the three-category denominator. A zero denominator has an undefined percentage, not 0%. The detailed `recommendations.csv` retains `raw_action`, `final_action`, `recommendation_status`, `review_flags`, and `diagnostic_flags`.
- `data/development/` and `data/test/` each contain the CSV results for all four independently evaluated MOD-18 models, restricted to each model's eligible stages (recommendations, memberships, rule activations, coverage, paired threshold changes, review flags and summaries), plus calibration, the rule registry and JSON validation/manifest metadata. Test results are descriptive, never used for policy selection.

**Figure labels only:** raw codes D0, D2, D6 and D10 display as **Stage 0, Stage 1, Stage 2 and Stage 3**, respectively. CSV values, file names, model inputs and the adjacent-stage recommendation logic retain their original coding. The Cognitive later rule referencing D0 performance is labelled “Stage 0 performance” inside the figures.

The compact presentation takes inspiration from Martínez et al. (2023), Martínez-Cid et al. (2024), Lee et al. (2020), Schoonderwoerd et al. (2021) and Liao et al. (2020), as considered within the 21-paper literature review. These studies support therapist-facing explanations, not our particular rule consequents or thresholds.

## Running the analysis

Once the proposed policy is documented and **frozen**, generate the combined folder. The `summary/` and `test/` directories contain the PNGs, and `data/` contains CSV/JSON results:

```powershell
& $Python .\python\MOD_18_fuzzy_recommendations.py `
  --modeling-data $ModelingData `
  --holdout-split $HoldoutSplit `
  --output-dir ".\outputs\MOD_18_results_boundary_v2" `
  --render-figure-set `
  --enable-locked-test-evaluation `
  --frozen-policy-id "ACTUAL_FROZEN_POLICY_ID" `
  --plot-threshold 0.20
```

The CLI requires a nonempty policy identifier but does **not** verify external approval. Do not use a placeholder to bypass the independent-test safeguard. Output directory must be absent or empty.

For development-only numerical exports and eight summary figures, use `--scope development --render-summary-plots` instead, with a separate output directory.
 
## Verification and scientific limits
Synthetic tests cover model restrictions, membership functions, conflict resolution, boundary-only Review, invalid-data status, participant-specific D0 matching without session chronology, numeric heatmaps, explicit rule labels, 80-figure counts, and scoped CSV exports. The official private-data pipeline must be executed separately. Rule counts and stage comparisons describe behaviour, not clinical recommendation accuracy. Repeated observations from the same participant are not independent people. The 5+5 rule libraries and boundary-only Review policy require therapist approval before clinical use; do not change them using held-out outcomes. The MOD-18 branch must not be merged without explicit approval and verified checks.
