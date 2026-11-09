# Auditory therapist walkthrough (read-only prototype)

This local browser viewer displays the eight precomputed Auditory observations prepared in thesis Stage 3.6. It does **not** run MOD-14 regression, MOD-16 SHAP or MOD-18 fuzzy inference. It cannot save a therapist decision or change VR settings.

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

## Screens C/D: numerical fuzzy explanations, using development data

The same browser page now includes C (Decision Tree–Mamdani) and D (Fuzzy C-Means–Sugeno). Original SHAP (A) and manual fuzzy rules (B) remain unchanged. **C/D do not produce Increase, Decrease or Maintain actions.**

### Study boundary

A/B currently uses two reserved-test cases, which were already examined. C/D uses two different participants from the frozen 26-person development cohort. The selectors and local JSON files are separate. Therefore, the four tabs are **not yet a matched-case comparison**. An eventual controlled therapist comparison should use development observations for all four approaches; do not use reserved-test cases to redesign the methods.

### Local data export

From the repository root in PowerShell, run the existing virtual environment and locate each original output directory by its unique file:

~~~powershell
function Find-One($name) {
    $matches = @(Get-ChildItem outputs -Recurse -File -Filter $name)
    if ($matches.Count -ne 1) {
        throw "Expected one $name; found $($matches.Count)"
    }
    return $matches[0]
}

$modelling = Find-One "modeling_data.csv"
$split = Find-One "participant_holdout_split.csv"
$tree = Find-One "numerical_rules.csv"
$mamdani = Find-One "output_term_manifest.json"
$activations = Find-One "mamdani_manifest.json"
$fuzzyRules = Find-One "tree_aligned_fuzzy_rules.csv"
$fcm = Find-One "cluster_manifest.json"
$sugeno = Find-One "zero_order_manifest.json"

$arguments = @(
    "prototype/auditory/build_cd_examples.py",
    "--modeling-data", $modelling.FullName,
    "--holdout-split", $split.FullName,
    "--mamdani-tree-dir", $tree.DirectoryName,
    "--mamdani-dir", $mamdani.DirectoryName,
    "--mamdani-activation-dir", $activations.DirectoryName,
    "--mamdani-fuzzy-dir", $fuzzyRules.DirectoryName,
    "--fcm-stage1-dir", $fcm.DirectoryName,
    "--sugeno-dir", $sugeno.DirectoryName,
    "--output", "outputs/auditory_cd_development_viewer.json"
)
python @arguments
if ($LASTEXITCODE -ne 0) { throw "C/D export failed." }
~~~

This local-only exporter uses MOD-14 to validate the 26/6 frozen split, then combines saved development tree rules, five-term Mamdani outputs and three-cluster Sugeno outputs. It validates participant-stage joins, crisp routes, eight Mamdani rule activations, centroid reconstruction, original-unit cluster centres and Sugeno memberships and numerical contributions. A two-dimensional PCA plot is computed **only for display** from the four training-scaled development features. PCA is not used to calculate fuzzy memberships. No reserved-test data are emitted or used for the projection.

The exporter refuses to overwrite an existing file, and writes a private, anonymised example JSON into the Git-ignored outputs directory. Do not commit that JSON.

Open prototype/auditory/index.html in the browser and select outputs/auditory_cd_development_viewer.json in the **C/D development examples** chooser. A/B continues to use its separate original JSON selector.

### Reading the visuals

**C** places the full decision tree in the main panel, with the session's crisp route highlighted. A second panel ranks the fuzzy rules that activate (including rules outside that crisp path), and a bottom plot shows the numerical Mamdani output centroid. The selected five-term vocabulary is provisional and must be labelled. Unavailable remains Unavailable.

**D** shows the 104 development points and three learned centres in a labelled PCA plot, highlights the current session, and displays all three fuzzy membership percentages. Selecting a cluster reveals the original-unit four-feature comparison. The bottom calculation lists each membership-times-constant contribution and the resulting Sugeno score. Membership is similarity to a multivariable profile, not a clinical category probability.

Both screens preserve the independent therapist judgement controls. The fuzzy model score is not an automatic treatment recommendation.

### Focused checks

~~~powershell
node --test ".\prototype\auditory\viewer.test.cjs"
node --test ".\prototype\auditory\viewer-cd.test.cjs"
~~~

The new C/D validation has 19 synthetic contract tests. It uses native browser SVG and requires no extra frontend dependencies. The real-data export and browser display must still be checked in the local thesis environment. Because the two fuzzy methods were fitted on development data and imitate the original regression, the displays are not independent clinical performance evidence.
