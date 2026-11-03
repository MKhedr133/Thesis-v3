# Scientific integrity audit — modelling and explainability

**Audit date:** 3 November 2026. **Basis:** static source inspection of `main` (MOD-09/14/15) and the unmerged MOD-16/17/18 branches. This is a scientific claim/validation audit, not a change to models or therapy policy. No participant data or private research outputs were accessed.

## 1. Evidence labels and inspected versions

- **Code safeguard observed:** the inspected source implements a check or boundary; this audit did not execute that code.
- **Known scientific limitation:** a statement is stronger than the design or evidence supports.
- **Open evidence gap:** further research data, clinical judgement, execution, or temporal provenance are needed. This is not automatically a software defect.
- **Executed test result:** use only when a recorded run confirms execution. Existing CI on PRs #31 and #34 passed 367 tests and 39 subtests, but not the unmerged MOD-16/17/18 code or official participant-data analysis.

Inspected sources: [MOD-09](../python/MOD_09_participant_holdout_split.py), [MOD-14](../python/MOD_14_direct_linear_regression_evaluation.py), [MOD-15](../python/MOD_15_visual_auditory_alternative_models.py), [MOD-16 / PR #25](https://github.com/MKhedr133/Thesis-v3/blob/feat/mod16-independent-linear-shap/python/MOD_16_linear_shap.py), [MOD-17 / PR #27](https://github.com/MKhedr133/Thesis-v3/blob/feat/mod17-compact-fuzzy-rules/python/MOD_17_compact_fuzzy_rules.py), [MOD-18 / PR #29](https://github.com/MKhedr133/Thesis-v3/blob/feat/mod18-prospective-fuzzy-recommendations/python/MOD_18_fuzzy_recommendations.py). These explanation/policy PRs are unmerged; the general task guide is proposed in PR #36.

## 2. Source-verified safeguards

| ID | Evidence from code | What it supports |
|---|---|---|
| C1 | MOD-09 creates a participant-level age-stratified split: 26 development (13 Young, 13 Old) and 6 reserved test (3 Young, 3 Old), using a fixed seed and participant IDs/age group rather than outcomes. | Split design; do not regenerate IDs after examining outcomes. |
| C2 | MOD-14 validates the 26/6 split, checks participant disjointness, fits separate frozen-registry MLR models on development rows and applies them to test rows without refitting. | A fitting boundary in code, not independent proof of the original numerical results. |
| C3 | MOD-15 uses GroupKFold by participant to evaluate Visual and Auditory alternatives with condition-specific predictors. | Grouped development CV rather than leakage-prone trial splitting. |
| C4 | MOD-14 computes participant-balanced MAE but ordinary **pooled** R². It retains continuous stage predictions using 0/1/2/3 to encode D0/D2/D6/D10. | Different metric aggregation schemes; R² does not mean percentage classification accuracy. |
| C5 | MOD-16 builds Independent Linear SHAP references from eligible development observations at each **known assigned stage**, checks no test participant enters the background, and verifies exact linear reconstruction. | Retrospective, within-stage attributions, not causal or prospective stage-selection effects. |
| C6 | MOD-17 uses development-only fuzzy anchors and additive continuation beyond them to reconstruct existing frozen MLR predictions. | Fidelity to the fitted equation, not model validity or clinical benefit. |
| C7 | MOD-18 calibrates stage-specific memberships on development observations; distinguishes Increase, Decrease, Maintain, boundary Review and invalid-data Unavailable. It checks for a unique participant-matched Cognitive D0 record. | Policy routing and baseline provenance, not proof that D0 was temporally available or any action was safe. |

## 3. Adversarial scientific claim checks

| Scenario | Expected interpretation or rejection | Audit result |
|---|---|---|
| A. One participant occurs in both development and test. | Reject; MOD-14 checks the participant boundary. | Code safeguard observed. |
| B. Trials from one person are divided between CV train and validation. | Reject; MOD-15 groups folds by participant. | Code safeguard observed. |
| C. A model, threshold or figure choice changes after observing test outcomes. | Cannot call the reused test evaluation untouched independent validation. | Methodological constraint; decision history needs independent records. |
| D. Held-out observations enter the MOD-16 SHAP reference. | Reject; background must comprise same-stage development observations. | Code safeguard observed. |
| E. Mean signed same-stage SHAP values are near zero. | This follows from coefficient centring within the reference group; it does not imply that predictors are irrelevant. | Mathematical consequence of the declared baseline. |
| F. The stage-specific MOD-16 plot explains which **unknown future** stage to assign. | Unsupported: reference selection uses the **known current assigned** stage. | Known retrospective scope. |
| G. Correlated variables have large Independent SHAP values. | Do not call them unique causal contributions; dependence-aware attribution is another method. | Interpretation limitation. |
| H. MOD-17 reconstruction error is zero. | Supports explanation fidelity to the MLR, not predictive accuracy, therapist comprehension or rehabilitation benefit. | Interpretation limitation. |
| I. Cognitive D0 is recorded after a later-stage Cognitive trial. | MOD-18's current unique-baseline lookup passes even though that D0 value was not available at the historical decision time. | Open temporal-readiness gap for prospective claims. |
| J. An input lies outside MOD-18 development calibration anchors. | An OUTSIDE_CALIBRATION diagnostic can accompany a candidate action; it is not an automatic Review/Unavailable veto. | Implemented policy, not established clinical safety. |
| K. The user supplies a nonempty `--frozen-policy-id`. | CLI accepts a nonempty identifier but cannot verify approval or real methodological freeze. | Documented governance gap. |
| L. MOD-18 rules are revised after earlier held-out Cognitive-later outcomes were inspected. | Subsequent evaluation on those six participants is **exploratory**, not untouched independent validation. A new unseen cohort is required for confirmation. | Confirmed in PR #29 history. |
| M. MOD-18 exports 72 individual test cards. | The cards represent repeated stages of six reserved participants, not 72 independent people. No proof of clinical usefulness. | Interpretation limitation. |
| N. High test R² or small MAE means the policy is safe or predicts treatment benefit. | Unsupported: predictive, inferential, explanatory and intervention claims need distinct evidence. | Claim rejected; preserve actual metric definitions. |

The table is a set of **reasoned counterexamples grounded in source inspection**. It is not a log of executed adversarial test cases.

## 4. Gates before future scientific claims

**G1 — Frozen model verification.** Preserve the MOD-09 split, established Visual/Auditory/Cognitive specifications, predictor names, coefficients and metrics. Before claiming reproduced research results, inspect actual versioned input files, fitted output tables and the authorized local run. A passing synthetic CI run is insufficient.

**G2 — MOD-16 and MOD-17 verification.** Run the relevant synthetic tests against the committed PR versions, then execute the official private-data workflow when authorized. Check regression prediction agreement, participant boundaries, stage eligibility, correlation caveats, complete explanation coverage and numerical reconstruction. Assess therapist-facing comprehension separately; figure legibility is not evidence of clinical benefit.

**G3 — MOD-18 temporal readiness.** Before describing outputs as **prospective next-session recommendations**, document which predictors and D0 baselines would have been available at the decision time. The current `baseline_status_for` checks that a same-participant Cognitive D0 record exists, **not chronology**. If temporal availability cannot be established, describe the analysis as retrospective. This is a proposed evidence requirement, not authorization to redesign the existing rules.

**G4 — Policy freeze and evaluation integrity.** Record a genuinely approved policy version, rules, thresholds, calibration basis, date and responsible reviewers outside the CLI string guard. Do not tune on reserved outcomes. Since the MOD-18 Cognitive-later policy was revised after viewing some held-out results, describe later evaluation of that policy on those people as exploratory. A new, unseen cohort is needed for confirmatory validation.

**G5 — Human/clinical validity.** Obtain therapist and supervisor input on rule rationale, uncertainty, out-of-calibration behavior, unavailable data, action boundaries and human override. Attributions and fuzzy memberships are not causal effects or clinical advice.

## 5. Verification scope and decision

**Performed:** Inspected GitHub code, key functions and PR research notes; checked the logical relation between grouped CV, frozen-model fitting, retrospective explanations and candidate adaptation; drafted 14 falsification/interpretation scenarios.

**Not performed:** No Python tests, CI workflows, official research-data execution, private-participant numerical inspection, SHAP feature rankings, clinician study or prospective action evaluation for this audit. The unmerged MOD-16/17/18 test suites have not been verified as passing in this work.

**Decision:** Keep the frozen models and existing methods unchanged. Record temporal availability, the documented policy freeze and exploratory status as explicit MOD-18 research gates. Report controls, limitations and unknowns separately; never convert a source inspection into a claim of scientific verification.
