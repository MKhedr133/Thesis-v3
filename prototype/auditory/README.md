# Auditory therapist walkthrough (read-only prototype)

This local viewer supports four Auditory explanations: SHAP (A), manual fuzzy rules (B), Decision Tree–Mamdani (C), and Fuzzy C-Means–Sugeno (D). Model inference for C/D is prepared locally, not fitted in the browser. The viewer cannot save therapist decisions or change VR settings.

## Current: one local file for all four methods (P06/P07)

The unified viewer uses **one JSON upload** and one shared participant/stage selector for A–D. P06 and P07 were originally reserved for testing, but their observations have **already been inspected** during earlier A/B interface development. Therefore these are **exploratory interface examples**, not untouched independent validation. The frozen Mamdani/Sugeno models stay fitted on 26 development participants; no model selection, modification, or test-performance metrics are performed.

The A/B source remains unchanged. The new exporter reads the existing local A/B JSON, checks it against canonical P06/P07 Auditory measurements and the original regression coefficients, and performs numerical C/D inference with the saved CART, fuzzy-rule partitions, five-term Mamdani output and three-cluster Sugeno constants. The PCA cloud still contains **104 development observations**; P06/P07 are separately projected onto it, without refitting the PCA or fuzzy model.

### Generate the unified file

If necessary, first prepare the original file by following **Legacy A/B-only preparation** below. Then execute in PowerShell from the thesis repository root:

    function Find-One($name) {
        $matches = @(Get-ChildItem outputs -Recurse -File -Filter $name)
        if ($matches.Count -ne 1) { throw "Expected one $name, found $($matches.Count)" }
        return $matches[0]
    }
    $modeling = Find-One "modeling_data.csv"
    $split = Find-One "participant_holdout_split.csv"
    $tree = Find-One "numerical_rules.csv"
    $mamdani = Find-One "output_term_manifest.json"
    $activations = Find-One "mamdani_manifest.json"
    $fuzzyRules = Find-One "tree_aligned_fuzzy_rules.csv"
    $fcm = Find-One "cluster_manifest.json"
    $sugeno = Find-One "zero_order_manifest.json"
    $argsList = @(
        "prototype/auditory/build_abcd_cases.py",
        "--legacy-json", "outputs/auditory_cases_interface_viewer.json",
        "--modeling-data", $modeling.FullName,
        "--holdout-split", $split.FullName,
        "--mod14-output-dir", "outputs/MOD_14_direct_linear_regression_evaluation",
        "--mamdani-tree-dir", $tree.DirectoryName,
        "--mamdani-dir", $mamdani.DirectoryName,
        "--mamdani-activation-dir", $activations.DirectoryName,
        "--mamdani-fuzzy-dir", $fuzzyRules.DirectoryName,
        "--fcm-stage1-dir", $fcm.DirectoryName,
        "--sugeno-dir", $sugeno.DirectoryName,
        "--output", "outputs/auditory_abcd_p06_p07_exploratory.json"
    )
    python @argsList
    if ($LASTEXITCODE -ne 0) { throw "Unified A–D export failed." }

The exporter refuses to overwrite an existing file. Keep the resulting participant-level JSON private under the Git-ignored outputs directory; do not commit it.

Open prototype/auditory/index.html and choose outputs/auditory_abcd_p06_p07_exploratory.json in the **single** file picker. Select P06 or P07 and the completed stage once; tabs A–D show matched observations. An older A/B-only JSON is still accepted for A/B, but will not populate C/D.

**A:** original stage-specific SHAP waterfall. **B:** original manually specified rules. **C:** full eight-leaf CART with highlighted crisp path, activated overlapping fuzzy rules and five-term Mamdani centroid. **D:** three FCM centres and the selected session on a development-only PCA overview, membership bars, original-unit feature comparisons and weighted Sugeno score. C/D numerical scores are not adaptation recommendations.

### Verification

Run the three viewer JavaScript test files using Node, and both Python export test files using unittest. Then generate the actual JSON and inspect all four browser tabs. The independent therapist-judgement fields remain unsaved.

---

## Legacy A/B-only preparation (optional)

## Prepare participant labels locally

The Stage 3.6 JSON uses Case A/B. The viewer requires a separate local JSON copy with a `ParticipantId` field so no private case-to-participant mapping is committed to Git. Run this in **Windows PowerShell** at the repository root. It derives the same deterministic selection from the stage-specific source.

```powershell
@'
import csv
import json
from collections import Counter
from pathlib import Path

shap = Path("outputs/MOD_16_linear_shap_stage_specific_20261101_180912/auditory/attributions_test.csv")
source = Path("outputs/auditory_cases_interface.json")
target = Path("outputs/auditory_cases_interface_viewer.json")

with shap.open(encoding="utf-8-sig", newline="") as file:
    participants = sorted({r["participant_id"] for r in csv.DictReader(file)})
if len(participants) != 6:
    raise ValueError("Expected exactly six reserved-test participants.")

with source.open(encoding="utf-8-sig") as file:
    records = json.load(file)
if not isinstance(records, list) or len(records) != 8:
    raise ValueError("Expected eight source observations.")

mapping = {"A": participants[0], "B": participants[1]}
for record in records:
    record["ParticipantId"] = mapping[record["Case"]]
if Counter(row["ParticipantId"] for row in records) != {
    mapping["A"]: 4, mapping["B"]: 4
}:
    raise ValueError("Unexpected participant counts.")

temporary = target.with_suffix(".json.tmp")
with temporary.open("w", encoding="utf-8") as file:
    json.dump(records, file, indent=2, ensure_ascii=False)
with temporary.open(encoding="utf-8") as file:
    check = json.load(file)
if not isinstance(check, list) or len(check) != 8:
    raise ValueError("Viewer JSON readback failed.")
temporary.replace(target)
print("Viewer JSON prepared:", target)
'@ | python -
```

## Open the prototype

From the repository root:

```powershell
Invoke-Item ".\prototype\auditory\index.html"
```

In the page, choose `outputs\auditory_cases_interface_viewer.json` using the local file selector. The viewer validates all eight observations, including SHAP numerical reconstruction (absolute tolerance `1e-8`), four feature identities, both thresholds, rule IDs and status/action consistency. It never sends the file to a server.

Select the participant and completed difficulty stage, then switch between **SHAP** and **Fuzzy rules**. The waterfall uses therapist-facing feature names but retains the original measurement IDs and signed contributions internally.

The **Fuzzy rules** tab shows the strongest positively activated rule(s) as complete IF–THEN statements, with activation numbers available under **All rules activations**. The main suggestion is a **threshold-free presentation summary**, not the final action of the frozen MOD-18 candidate policy. If no rule has positive activation, it displays Maintain. Equal strongest Increase and Decrease rules display Mixed indications. A strongest directional rule pointing beyond Stage 0 or Stage 3 displays Review. An unavailable recorded recommendation displays Unavailable.

The underlying MOD-18 outcomes for thresholds 0.10 and 0.20, support values and diagnostic flags remain intact in the local JSON but are not displayed. They may disagree with the summary shown in the interface; neither policy threshold is silently selected or tuned. For scientific reporting, describe the main value as a **rule-level indication**, not a validated adjustment or the original MOD-18 final action.

The ten antecedent definitions are copied from the maintained Auditory MOD-18 rule registry and are not recalculated from SHAP. Therapist choice and rationale fields are unsaved, and selecting another observation clears them.

## Focused synthetic checks

If Node.js is available, run from the repository root:

```powershell
node --test ".\prototype\auditory\viewer.test.cjs"
```

No `npm install` is required. The browser viewer itself needs only a modern browser.

## Scientific and privacy limits

- Participants P06 and P07 are selected from the reserved-test group. The viewer is a **demonstration**, not untouched independent validation of an interface developed after examining these cases.
- Regression output is retrospective and continuous; it is not automatically rounded or converted to the optimal future-session stage.
- SHAP values represent contributions to the fitted model relative to its stage-specific development-only baseline, not causal effects.
- MOD-18 is a candidate policy, not a clinical prescription. Review and Unavailable remain distinct and the clinician retains final judgement.
- Original data, the participant mapping and locally generated JSON are ignored by Git under `outputs/`. Never commit or publicly host participant-level data.

## Project lifecycle

Scoped issue: #40. This implementation is deliberately separate from the MOD-16 and MOD-18 model branches and contains no changes to regression or fuzzy coefficients, features, splitting, thresholds or rule definitions.

