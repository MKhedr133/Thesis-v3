# Native MATLAB FE-01 port

This folder contains the independent MATLAB implementation of the completed
FE-01.1--FE-01.15 extraction pipeline. Python remains the reference for
cross-language parity checks. The public functions live in the `+fe01`
package, and the class-based tests are in `tests/Fe01Test.m`.

## Run the MATLAB tests

Run these commands in MATLAB after changing `thesisRoot` to the local thesis
checkout:

```matlab
thesisRoot = "D:\TU\MScRobotics\Thesis_files\Rebuild_v3\Thesis";
addpath(fullfile(thesisRoot, "matlab"));
testResults = runtests(fullfile(thesisRoot, "matlab", "tests", "Fe01Test.m"));
disp(table(testResults));
```

The test class uses synthetic trackers and temporary output directories. The
optional parity test is skipped when the approved Python trial CSV is absent.

## One-trial smoke extraction

```matlab
thesisRoot = "D:\TU\MScRobotics\Thesis_files\Rebuild_v3\Thesis";
addpath(fullfile(thesisRoot, "matlab"));
result = fe01.run_full_extraction( ...
    "D:\TU\MScRobotics\Thesis_files\python\EDA_v2\scripts\inputs\performance_1.csv", ...
    "D:\TU\MScRobotics\Thesis_files\data\raw\data_full", ...
    "D:\TU\MScRobotics\Thesis_files\python\EDA_v2\scripts\inputs\CorrectLists.txt", ...
    fullfile(thesisRoot, "outputs", "feature_extraction_v2_matlab_smoke"), ...
    "Limit", 1, ...
    "MentalDemandPath", ...
    "D:\TU\MScRobotics\Thesis_files\python\EDA_v2\scripts\inputs\mental_demand.xlsx");
disp([height(result.trialFeatureTable), height(result.productGrabTable), height(result.qcTable)]);
```

## Full MATLAB extraction

After the one-trial tables, warnings, manifest, and reach values have been
compared with the Python smoke output, use a separate output directory for
the complete run:

```matlab
result = fe01.run_full_extraction( ...
    "D:\TU\MScRobotics\Thesis_files\python\EDA_v2\scripts\inputs\performance_1.csv", ...
    "D:\TU\MScRobotics\Thesis_files\data\raw\data_full", ...
    "D:\TU\MScRobotics\Thesis_files\python\EDA_v2\scripts\inputs\CorrectLists.txt", ...
    fullfile(thesisRoot, "outputs", "feature_extraction_v2_matlab"), ...
    "MentalDemandPath", ...
    "D:\TU\MScRobotics\Thesis_files\python\EDA_v2\scripts\inputs\mental_demand.xlsx");
```

The MATLAB port keeps the Python schemas and output filenames, writes missing
numeric values as blank CSV cells, and creates thirteen invisible `Agg`-style
descriptive plots under the selected output directory. No participant-detail
metadata is cached across trials and no parallel processing is used.
