# Thesis task workflows

This guide complements [CONTRIBUTING.md](../CONTRIBUTING.md). It defines task-specific research and communication procedures, not new modelling decisions. Before each substantial task, inspect the current repository, project sources, supervisor decisions and pending pull requests.

## Rules shared across tasks

- Preserve independent Visual, Auditory and Cognitive modelling, including separate Cognitive primary and later specifications. Keep established predictor sets, frozen coefficients, metrics and D0/D2/D6/D10 coding (numeric stages 0/1/2/3).
- The study includes 32 participants: **26 development** and **6 reserved for independent testing**. Use participant-grouped resampling; never perform random trial-level splitting. All model selection, preprocessing choices, calibration, SHAP backgrounds, rule and threshold decisions must use development data alone.
- Never tune on held-out outcomes. Where previous test results influenced later policy design, report subsequent evaluation as **exploratory**, not untouched independent validation. The MOD-18 Cognitive-later policy has this limitation.
- Distinguish observed associations, model attributions, retrospective descriptions and candidate next-session actions. These are not causal explanations, verified clinical benefits or therapist-approved instructions.
- Do not upload private participant data, unpublished numerical outputs or credentials to public services, GitHub, or external research connectors.

## 1. Literature research and citation verification

**When:** An evidence claim, citation check, methodology comparison or literature review.

1. Define the question, population/task, relevant condition and inclusion criteria. Check supplied project papers and reading notes first; do not infer findings from titles or abstracts alone.
2. Search original studies and metadata as needed. SciSpace supports semantic discovery; Sider Scholar/OpenAlex supports identifiers and source records; Consensus supports broader searches, but its individual search results must be fetched before citation.
3. Verify the DOI, publication, methods, actual outcomes and limitations against the original paper. Distinguish primary findings, review summaries, preprints and your own inference.
4. Compare evidence by participants, task, models, explanation format, evaluation measures, findings and limitations; include contradictory or missing evidence.
5. Deliver supported claims with citations and explicit relevance to the thesis. Do not interpret generic XAI preference studies as evidence that a particular therapist interface improves clinical decisions.

**Acceptance check:** Each empirical claim has a verifiable original source; unsupported claims are identified. Example: Aas, Jullum and Løland (2021), DOI 10.1016/j.artint.2021.103502, addresses Shapley explanations when features are dependent. It does not require a change to the thesis's Independent Linear SHAP approach.

## 2. Modelling, statistics and validation

**When:** Data preparation, prediction metrics, cross-validation, regression or alternative-model comparison.

1. Read the relevant scripts and tests, especially MOD-09 split, MOD-13 model registry, MOD-14 evaluation and MOD-15 alternative models. Check source-column definitions and the frozen participant manifest.
2. Define the outcome, predictors, target coding, eligible participants/trials, missingness handling and metric formula. Separate predictive MAE, R² and explicitly defined tolerance accuracy from coefficient inference or attribution.
3. Use development participants only for model/feature choice, grouped validation, transformations, threshold selection and hyperparameters. Preserve the six reserved participants for the stated evaluation boundary; no hidden trial-level leakage.
4. Inspect assumptions, scaling, repeated trials, condition-specific design, units and numerical stability. Never introduce imputation, cross-condition pooling or revised coefficients without an approved methodological decision.
5. Present validation results with denominators, uncertainty, training/development versus test distinctions and caveats. If held-out outcomes affected a revision, label evaluation exploratory.
6. Run synthetic tests and, when authorized, execute the official pipeline on local research data. Record exact commands, input provenance and checked outputs.

**Acceptance check:** The participant split and regression contracts remain intact; formulas, metrics, units and execution evidence are reproducible. A passing CI run is not scientific verification.

## 3. Explainability and candidate adaptations

**When:** SHAP figures, fuzzy linguistic explanations, prospective difficulty suggestions or therapist interpretation.

- **MOD-16 (pending PR #25):** Independent Linear SHAP decomposes each frozen continuous regression prediction relative to a development-only reference for the **known assigned stage**. Verify the stage reference, signed contributions, additive reconstruction and agreement with the original prediction. This answers a retrospective within-stage question; it is not conditional SHAP, causal influence or prospective stage selection.
- **MOD-17 (pending PR #27):** Its Low/Medium/High fuzzy description reconstructs frozen MLR predictions. Verify development-only anchors, exact fidelity, stage/model boundaries and treatment of values beyond anchors. Linguistic membership is not a treatment recommendation.
- **MOD-18 (pending draft PR #29):** A separate candidate prospective system proposes Increase, Decrease or Maintain. Boundary **Review** and data **Unavailable** are non-action statuses. Verify model-specific stages, participant-specific Cognitive D0 comparisons and whether a baseline would genuinely be available when acting. After prior held-out Cognitive-later outcomes influenced revisions, subsequent held-out assessment is exploratory and needs a new unseen cohort for confirmatory validation.
- **Human evaluation:** Test explanation fidelity and stability separately from whether therapists understand it, can anticipate model behaviour or use it responsibly. Do not infer effectiveness from readable figures.

**Acceptance check:** Every output states model/condition, assigned stage, data boundary, reference, units and scope. No model refitting without approval; no attribution interpreted as causal or clinical guidance.

## 4. GitHub development and repository selection

**When:** Build, debug, simplify, test, review, or adopt a dependency.

1. Inspect `MKhedr133/Thesis-v3`, code, callers, tests, requirements, open issues, PRs and the correct base branch (including stacked branches). Prefer existing standard/scientific tools. If a new repository could materially help, compare scientific fit, current maintenance, licence, security, compatibility, documentation, tests, adoption, effort and zero-additional-cost use; stars are supporting evidence, not proof.
2. Create/update a scoped issue with acceptance criteria, non-goals and scientific/privacy constraints. Branch from the appropriate base, make focused Conventional Commits, and avoid unnecessary classes, wrappers, flags or packages.
3. Review code changes and test contracts. Run available synthetic checks and record actual logs, test counts and failures. A workflow that forces lint success with `--exit-zero` is an audit, not a clean lint pass.
4. Open a linked PR with diff, tests, limitations and research impact. Address feedback and preserve participant grouping, model invariants and reproducibility.
5. Merge **only** after required checks pass and the user explicitly authorizes the merge. Do not delete branches, deploy or change protected research methods without approval.

**Acceptance check:** A focused reviewed diff and traceable issue, commit, tests and PR. Report implementation, CI, official local execution and scientific verification separately.

## 5. Academic writing and supervisor material

**When:** A chapter, literature synthesis, methods note, supervisor report, substantive documentation or revision.

1. Identify audience, purpose and primary sources. Make the account self-contained: project context, scientific aim, methods, evidence/results, interpretation, limitations and next steps.
2. Verify original citations, numerical tables, figure values, terms, units and model definitions. Distinguish verified results from hypotheses and never invent citations or missing outcomes.
3. Draft using Natural Writing MCP v2 DCR: call `review_text` with `technical` context, correct genuine editorial problems, and avoid indiscriminate rewrites of scientific terminology. For a revision, call `check_preservation` against the original, then re-review after substantial corrections. Review documents longer than **6,000 characters** in coherent sections.
4. Use connected Google Drive/Docs to find/create/edit native documents when requested, verifying the destination and completed operation. Do not claim that a document or check exists unless the tool confirms it.

**Acceptance check:** Natural, readable technical writing, traced claims, intact citations and scientific qualifications, and reported writing/preservation checks.

## 6. Figures, slides and scientific communication

**When:** Regression evaluation figures, SHAP waterfalls, fuzzy plots, supervisor slides or other visual artifacts.

1. Match the target format to the communication need: existing Matplotlib for reproducible PNGs, editable presentation tools for slides, or an interactive library only if interaction is essential.
2. Keep Visual, Auditory and Cognitive findings distinct. Label stage mapping, model identity, training/development versus test, predictor units, sample sizes as participants or trials, baselines and meaningful uncertainty.
3. Trace plotted values to verified data. Label illustrative synthetic values explicitly. Do not manufacture private participant results or present candidate actions as clinician-endorsed advice.
4. Inspect the rendered result for legibility, missing labels, misleading scales or ambiguous grouping. Check figure-output contracts and use tests where available.

**Acceptance check:** A supervisor can understand the graphic's question, units, source, limitations and result without conflating test and training or independent participants and repeated trials.

## 7. Task routing and evidence report

| Task | Inspect/use first | Evidence to provide |
|---|---|---|
| Verify a literature claim | Supplied original paper; scholarly source record | Verified finding, DOI/citation, limitation |
| Evaluate a prediction | MOD-09/13/14/15 and split manifest | Formula, grouped test, metrics and provenance |
| Explain a model | Frozen MLR and MOD-16/17/18 branch | Model/stage reference, fidelity check, caveats |
| Change code or tests | GitHub issue, code/callers and CI | Branch, commits, reviewed diff, actual test outcome |
| Draft/revise a report | Project sources, scholarly tools, Natural Writing | Complete reviewed text, preservation status |
| Create a figure or slide | Numerical source and required output format | Inspected artifact, units and data boundary |

**Status vocabulary:** *Implemented* means files/code exist; *CI passed* means a named test run succeeded; *official analysis executed* requires evidence from the real local inputs; *scientifically verified* requires verified outputs and justified interpretation. These are separate milestones.

This document is an on-demand routing guide, not an always-running agent or a grant of permission to modify established models, use reserved outcomes for tuning, or merge pending PRs.