# Auditory therapist walkthrough (read-only prototype)

This local browser viewer displays the eight precomputed Auditory observations prepared in thesis Stage 3.6. It does **not** run MOD-14 regression, MOD-16 SHAP or MOD-18 fuzzy inference. It cannot save a therapist decision or change VR settings.

## Prepare participant labels locally

The Stage 3.6 JSON uses Case A/B. The viewer requires a separate local JSON copy with a `ParticipantId` field so no private case-to-participant mapping is committed to Git. Run this in **Windows PowerShell** at the repository root. It derives the same deterministic selection from the stage-specific source.

```powershell
& {
    $ErrorActionPreference = "Stop"

    $shapPath = ".\outputs\MOD_16_linear_shap_stage_specific_20261101_180912\auditory\attributions_test.csv"
    $sourcePath = ".\outputs\auditory_cases_interface.json"
    $targetPath = ".\outputs\auditory_cases_interface_viewer.json"

    $rows = @(Import-Csv -LiteralPath $shapPath)
    $participants = @($rows.participant_id | Sort-Object -Unique)

    if ($participants.Count -ne 6) {
        throw "Expected six reserved-test participants."
    }

    $mapping = @{
        A = $participants[0]
        B = $participants[1]
    }

    $records = ConvertFrom-Json -InputObject (
        Get-Content -LiteralPath $sourcePath -Raw
    )
    if ($records -isnot [array] -or $records.Count -ne 8) {
        throw "Expected a JSON array with eight observations."
    }

    foreach ($record in $records) {
        $key = [string]$record.Case
        if (-not $mapping.ContainsKey($key)) {
            throw "Unexpected case key."
        }
        $record | Add-Member -NotePropertyName ParticipantId -NotePropertyValue $mapping[$key] -Force
    }

    ConvertTo-Json -InputObject $records -Depth 15 |
        Set-Content -LiteralPath $targetPath -Encoding UTF8

    $check = ConvertFrom-Json -InputObject (
        Get-Content -LiteralPath $targetPath -Raw
    )
    if ($check -isnot [array] -or $check.Count -ne 8) {
        throw "Local JSON readback failed."
    }

    Write-Output "Viewer data prepared: $targetPath"
    Write-Output "Observations: $($check.Count)"
    Write-Output "Participants: $($mapping.A), $($mapping.B)"
}
```

## Open the prototype

From the repository root:

```powershell
Invoke-Item ".\prototype\auditory\index.html"
```

In the page, choose `outputs\auditory_cases_interface_viewer.json` using the local file selector. The viewer validates all eight observations, including SHAP numerical reconstruction (absolute tolerance `1e-8`), four feature identities, both thresholds, rule IDs and status/action consistency. It never sends the file to a server.

Select participant and completed stage, then switch between Regression (SHAP) and Fuzzy recommendation. Both threshold outputs are shown; there is no implicitly approved threshold. Therapist choice and rationale fields are unsaved, and selecting another observation clears them.

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
