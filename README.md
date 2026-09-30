# Thesis v3

[![Python tests](https://github.com/MKhedr133/Thesis-v3/actions/workflows/python-tests.yml/badge.svg)](https://github.com/MKhedr133/Thesis-v3/actions/workflows/python-tests.yml)

Research software for an MSc Robotics thesis about difficulty prediction in a VR supermarket task.

The project studies whether behavior, performance, and subjective rating can be used to predict task difficulty. The long term goal is to use these measurements to support a therapist when choosing a suitable difficulty for a participant.

The repository contains the Python feature extraction and modelling code, automated tests, and the GitHub development workflow used during the thesis.

> **Important:** predicting the difficulty that matches a participant's measurements is not the same as deciding which difficulty is best for that participant. The final recommendation method is still to be defined. The therapist always makes the final decision.

## Main features

- Data from 32 participants in a VR supermarket task
- Three separate conditions: Visual, Auditory, and Cognitive
- Four difficulty stages: D0, D2, D6, and D10
- Behavioral feature extraction from VR tracker data
- Performance and mental demand measurements
- Mixed effects models for statistical analysis
- Participant based train and test split
- Participant bootstrap analysis
- Feature selection based on statistical association with difficulty
- Multiple linear regression for difficulty prediction
- Automated Python tests with GitHub Actions
- Participant data kept outside the GitHub repository

## Research objective

The practical goal is to use measurements from the VR supermarket task to support difficulty recommendations.

The planned system has two stages.

1. **Initial recommendation**

   A participant first completes D0. Measurements available after D0 can then be used to support an initial difficulty recommendation.

2. **Later adaptation**

   After the participant completes a higher difficulty stage, new measurements can be used to support a later decision about whether the difficulty should increase, decrease, or stay the same.

The current work focuses on predicting the difficulty stage.

The separate logic for deciding whether a difficulty is suitable for a participant has not yet been defined.

## Experimental design

| Item | Design |
| --- | --- |
| Participants | 32 |
| Participant groups | 16 younger, 16 older |
| Conditions | Visual, Auditory, Cognitive |
| Difficulty levels | D0, D2, D6, D10 |
| Difficulty coding | 0, 1, 2, 3 |
| Trials per participant | 12 |
| Total trials | 384 |
| Main outcomes | Relative performance and mental demand |
| Behavioral features | 8 |

The three conditions are analyzed separately because each condition has its own D0 baseline.

The difficulty stages are coded as:

```text
D0  = 0
D2  = 1
D6  = 2
D10 = 3
```

These numbers represent order only. They do not mean that the distance between the stages is equal.

`T1` to `T4` describe trial order. They do not describe difficulty.

Each participant completes several trials. These trials are related because they come from the same person. The analysis therefore keeps participant identity together during modelling, data splitting, cross validation, and bootstrap analysis.

## Measurements

### Relative performance

The task contains 20 products.

Performance percentage is calculated as:

```text
correct products / 20 × 100
```

Relative performance is calculated within the same participant and condition:

```text
D0 performance percentage - current performance percentage
```

This means:

- positive value: performance became worse than D0
- zero: no change from D0
- negative value: performance improved compared with D0

At D0, this value is always zero. It therefore does not provide useful new information for the first recommendation after D0.

### Mental demand

Mental demand is recorded after each trial using a score from 0 to 10.

Mental demand and performance are treated as separate measurements.

A participant may report higher mental demand without showing the same change in task performance.

### Behavioral features

The feature extraction currently calculates eight behavioral measurements.

| Measurement | Unit |
| --- | --- |
| Typical time between correct product grabs | seconds |
| List recheck count | count |
| Total list recheck duration | seconds |
| Typical time to locate a target product | seconds |
| Typical time focused on irrelevant objects during search | seconds |
| Typical head turning while locating a product | degrees |
| Typical time to reach for a product | seconds |
| Reach path ratio | ratio |

The behavioral summaries use valid first time grabs of products that are on the shopping list.

Missing measurements remain missing. They are not changed to zero.

Real zero values remain zero when zero has a clear meaning. For example, a participant may have zero list rechecks.

Object focus comes from labels recorded by the VR system. It should not be described as physiological eye fixation.

## Current implementation

| Stage | Status |
| --- | --- |
| Python feature extraction | Implemented |
| Feature extraction tests | Implemented |
| Raw TMT-B modelling | Implemented |
| Participant train and test split | Implemented |
| Random effects comparison | Implemented |
| Participant bootstrap model comparison | Implemented |
| Final difficulty association analysis | Implemented |
| Difficulty prediction with multiple linear regression | Implemented |
| Bootstrap analysis for difficulty prediction | Implemented |
| Protected final test process | Implemented |
| Final recommendation method | Not yet defined |
| Population reference method | Deferred |
| Feature extraction threshold validation | Pending |
| MATLAB implementation | Deferred |
| Python and MATLAB comparison | Deferred |

Code being present in GitHub does not automatically mean that the official analysis was run.

Official analysis results are only treated as verified after the code has been run on the thesis data and the generated outputs have been checked.

## Repository structure

```text
Thesis-v3/
├── .github/
│   └── workflows/
│       └── python-tests.yml
├── python/
│   ├── 02_extract_selected_features.py
│   ├── MOD_02_random_effects_structure.py
│   ├── MOD_04_age_tmt_associations.py
│   ├── MOD_05_lopo_predictive_comparison.py
│   ├── MOD_09_participant_holdout_split.py
│   ├── MOD_10_training_random_effects_comparison.py
│   ├── MOD_11_training_bootstrap_comparison.py
│   ├── MOD_12_final_training_associations.py
│   └── MOD_13_difficulty_prediction.py
├── tests/
│   ├── test_02_extract_selected_features.py
│   ├── test_mod02_optimizer_selection.py
│   ├── test_mod09_participant_holdout_split.py
│   ├── test_mod10_training_random_effects_comparison.py
│   ├── test_mod11_training_bootstrap_comparison.py
│   ├── test_mod12_final_training_associations.py
│   └── test_mod13_difficulty_prediction.py
├── CONTRIBUTING.md
├── requirements-dev.txt
└── README.md
```

## Main Python modules

| Module | Purpose |
| --- | --- |
| `02_extract_selected_features.py` | Extract measurements from the VR data |
| `MOD_02_random_effects_structure.py` | Shared functions for mixed effects models |
| `MOD_04_age_tmt_associations.py` | Analyze Age and TMT-B effects |
| `MOD_05_lopo_predictive_comparison.py` | Compare candidate models using leave one participant out prediction |
| `MOD_09_participant_holdout_split.py` | Create the participant training and test split |
| `MOD_10_training_random_effects_comparison.py` | Compare random effects structures using training participants |
| `MOD_11_training_bootstrap_comparison.py` | Compare candidate models using participant bootstrap analysis |
| `MOD_12_final_training_associations.py` | Test the final associations between measurements and difficulty |
| `MOD_13_difficulty_prediction.py` | Predict difficulty stage using multiple linear regression |

## Requirements

The GitHub test environment uses Python 3.11.

The Python packages are listed in `requirements-dev.txt`:

```text
numpy
pandas
matplotlib
openpyxl
statsmodels==0.15.0
```

MATLAB is not required for the current Python workflow.

## Installation

This is a private repository. GitHub access is required.

Clone the repository:

```bash
git clone https://github.com/MKhedr133/Thesis-v3.git
cd Thesis-v3
```

Create a Python environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Usage

### Run the tests

Run all Python tests from the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

GitHub Actions runs the same test suite when code is pushed to `main` or when a pull request targets `main`.

Passing these tests means that the tested software behavior works as expected.

It does not mean that the official participant data were analyzed.

### Run feature extraction

Show the available options:

```bash
python python/02_extract_selected_features.py --help
```

Example full extraction:

```bash
python python/02_extract_selected_features.py \
    --performance path/to/performance_file \
    --raw-root path/to/raw_participant_data \
    --correct-lists path/to/CorrectLists.txt \
    --mental-demand path/to/mental_demand.xlsx \
    --run-full-extraction \
    --output-dir path/to/output_directory
```

Participant data are stored outside the repository, so the input paths must be provided when the script is run.

The extraction creates trial features, product grab information, quality control information, plots, and a run manifest.

### Run difficulty model development

Show the available MOD 13 commands:

```bash
python python/MOD_13_difficulty_prediction.py --help
```

Run model development using the training participants:

```bash
python python/MOD_13_difficulty_prediction.py develop \
    --modeling-data path/to/modeling_data.csv \
    --holdout-split path/to/participant_holdout_split.csv \
    --output-dir path/to/mod13_development
```

This stage uses only the training participants.

It performs participant bootstrap analysis and then fits the final regression equations using the training data.

Main outputs:

```text
regression_coefficients.csv
bootstrap_performance.csv
```

### Run the final test

The final test is protected so that it cannot be run by accident.

```bash
python python/MOD_13_difficulty_prediction.py final-test \
    --modeling-data path/to/modeling_data.csv \
    --holdout-split path/to/participant_holdout_split.csv \
    --coefficients path/to/regression_coefficients.csv \
    --output-dir path/to/mod13_final_test \
    --confirm-final-test
```

The `--confirm-final-test` option is required.

The models are evaluated on the test participants without fitting them again.

Main outputs:

```text
final_test_performance.csv
final_test_predictions.csv
```

After the final test has been inspected, the test results must not be used to change the model, predictors, preprocessing, or feature selection.

## Data and reproducibility

Participant data are not stored in this repository.

The repository contains:

- Python source code
- automated tests
- GitHub Actions configuration
- development documentation

The repository does not contain:

- participant data
- raw VR tracker files
- private credentials
- local environment files
- official analysis outputs unless they are explicitly approved for GitHub

The official thesis analyses are run in the local thesis environment.

GitHub tests and the official thesis analysis have different purposes.

```text
GitHub tests
→ check whether the code behaves as expected

Official local analysis
→ runs the code on the thesis data

Scientific verification
→ checks and interprets the generated results
```

These steps should not be treated as the same thing.

## Development workflow

Development follows a simple GitHub workflow.

1. Create or select a GitHub issue.
2. Define the task and acceptance criteria.
3. Create a branch from the latest `main`.
4. Make small and clear commits.
5. Open a pull request into `main`.
6. Run the automated tests.
7. Review the code changes.
8. Merge the pull request when the checks are complete.
9. Delete the branch after the merge.

Common commit prefixes are:

```text
feat:
fix:
test:
docs:
chore:
```

For small and focused pull requests, squash merge is preferred.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more details.

## Project records

GitHub stores the software and technical development history.

`PROJECT_STATE.md` stores the current methodological and research decisions.

Official analysis outputs are kept separately in the local thesis workspace.

```text
GitHub
→ code and development history

PROJECT_STATE.md
→ current research and methodology decisions

Local analysis outputs
→ official thesis results
```

This keeps software development separate from scientific conclusions.

## Remaining work

The main remaining work includes:

- interpret and report the difficulty prediction results
- decide how continuous predictions should be converted to D0, D2, D6, or D10
- define how the system should decide whether difficulty should increase, decrease, or stay the same
- define what makes a difficulty suitable for a participant
- validate important feature extraction thresholds
- complete MATLAB work if it is required for the final thesis

## Contributing

This is a private MSc thesis repository.

Changes should follow the workflow in [`CONTRIBUTING.md`](CONTRIBUTING.md).

GitHub Issues should be used to define larger implementation tasks before coding begins.

## Support

Use the GitHub issue tracker for software problems or implementation tasks.

Research and methodology decisions should be recorded in `PROJECT_STATE.md`.

## Author

**Mohamed Khedr**  
MSc Robotics

## License

No open source licence is currently included.

This repository should therefore be treated as private thesis work.
