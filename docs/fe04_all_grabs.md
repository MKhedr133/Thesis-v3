# FE-04 descriptive search and reach figures

The `--all-grabs` option in `python/09_extract_final_features.py` includes every valid product grab in the descriptive search and reach analysis. This includes repeats and products not on the shopping list. The search target is the product actually grabbed. The search window begins at the previous valid grab's start or at the end of a later completed list visit. For the first grab, the start of usable trial activity replaces the previous grab. Grabs with the same start time use the preceding distinct grab start. A usable reach requires logged focus on the grabbed product within that window and detectable hand movement. The original motion and path thresholds still apply. Missing focus leaves both measures missing.

The original extractor remains the default when the option is omitted. The FE-03 threshold sweep used first grabs of listed products and is unchanged. Its results describe that earlier definition, not sensitivity under the new one. FE-04 is for descriptive reporting and does not change frozen model inputs, the participant split or model evaluation.

## Regenerate the figures

Use the same official trial table, raw tracker directory, product catalogue and mental demand workbook used for the original full-cohort run. In PowerShell, set `$rawRoot`, `$catalogue` and `$mentalDemand` to those local paths, then run from the repository root.

```powershell
$trialTable = ".\inputs\performance_1.csv"
$featureDir = ".\outputs\fe04_all_grabs_01"
$figureDir = ".\outputs\fe04_descriptive_figures_01"

python .\python\09_extract_final_features.py `
  --performance $trialTable `
  --raw-root $rawRoot `
  --correct-lists $catalogue `
  --mental-demand $mentalDemand `
  --all-grabs `
  --output-dir $featureDir
if ($LASTEXITCODE -ne 0) { throw "The extraction did not finish." }

python .\python\11_plot_descriptive_features.py `
  (Join-Path $featureDir "trial_features.csv") $figureDir
if ($LASTEXITCODE -ne 0) { throw "The figure export did not finish." }
```

If the workbook is named `mental_demand.xlsx` and sits beside the product catalogue, the `--mental-demand` line can be omitted. Both output directories must be new. Keep `feature_extraction_manifest.json` beside `trial_features.csv` to let the plotter verify the extraction scope and CSV hash.

The figure directory contains three individual vector PDFs, a combined PDF, captions, grouped summaries and a plot manifest. Group summaries report available and missing trial counts separately. If an entire condition and difficulty group has no usable values, the plot marks `n = 0` and shows no median. Keep participant inputs and outputs local.

For a worked timeline, add `--worked-example PARTICIPANT SESSION TRACKER GRAB_NUMBER` to the extraction command and run `python python/plot_fe02_worked_example.py <worked_trace.json> <output.pdf>`. Use the 1-based row number in the valid-grab audit table for that trial, and choose a grab with a usable search and reach path.
