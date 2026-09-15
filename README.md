# Thesis v3

MSc Robotics thesis rebuild in Python and MATLAB, developed through small, reviewed, tested steps.

## Research goal

Use measurements available after baseline difficulty D0 in a VR supermarket task to recommend a suitable subsequent difficulty and explain that recommendation to a therapist. The therapist makes the final decision and may accept, change, or reject the recommendation. The system must not apply difficulty changes automatically.

Treat Visual, Auditory, and Cognitive as separate recommendation problems: each condition's D0 measurements support recommendations for that same condition. Do not pool the three D0 baselines or use later-difficulty measurements as inputs available at the original D0 decision point.

## Experimental context

The student-reported design includes 32 participants (16 younger and 16 older), seated task performance, and dominant-hand product collection. Each condition contains D0, D2, D6, and D10 trials in shuffled order. T1-T4 describe chronological order, not difficulty. D0 is the no-added-manipulation baseline within a condition.

These are reported design facts, not counts verified by running the rebuild. The final recommendation choices, meaning of suitable difficulty, D0 duration, predictor list, and evaluation design remain unresolved.

## Current measurement candidates

Feature definitions are complete for now; extraction rules and numerical thresholds still require validation.

| Feature | Trial measure | Unit |
| --- | --- | --- |
| Typical time between correct product grabs | Median interval between consecutive first-time on-list grabs; missing with fewer than two | Seconds |
| List recheck count | Valid list visits after the initial viewing; no per-product division | Count |
| Total list recheck time | Total duration of those rechecks; zero when there are none | Seconds |
| Typical time to locate the target product | Median interval from search start to first valid target focus | Seconds |
| Typical time focused on irrelevant objects during search | Median irrelevant-object focus duration within locating intervals | Seconds |
| Typical time to reach for a product | Median interval from detected reach onset to grab; onset detector unresolved | Seconds |
| How indirect the hand movement was | Median reach path length divided by straight-line endpoint distance | Dimensionless |
| Typical head turning while locating a product | Median accumulated 3-D headset angular path within locating intervals | Degrees |

Product-level summaries use qualifying first-time on-list grabs. Search may start after the previous qualifying grab; a later relevant completed list visit restarts it after that visit ends. It does not wait for the previous product's release. The first product uses usable trial activity start, adjusted for a relevant list visit.

Engine-labelled object focus is not a physiological eye fixation. Irrelevant focus includes NPCs and defensibly unrelated scene objects; it excludes products, shelves, the list, the cart, and invalid samples. The actual logged labels still need review. These features are measurement candidates, not an approved final model input list.

## Outcomes and open decisions

- Keep performance change and subjective mental demand separate. Performance deterioration is positive relative to the same participant's same-condition D0; mental demand is reported on a 0-10 scale after each trial.
- Preserve raw D0 total errors and the four component counts. Higher-difficulty error change is `errors_at_difficulty - errors_at_D0`; it is a later response, not a D0 predictor.
- No model, explanation method, recommendation target, or numerical extraction threshold has been selected. Historical defaults are not approved thesis thresholds.
- Supervisor feedback remains outstanding. Reported conceptual reviews do not establish validated extraction, patient benefit, or clinical effectiveness.

## Implementation workflow

ChatGPT supports research planning and theory; Codex implements accepted decisions in Python and MATLAB.

1. Define the measurement, then agree and validate event thresholds and preprocessing before rewriting the extractor.
2. Implement one approved, testable microstep on a separate implementation branch. Identify its decision source, objective, inputs, outputs, files, and expected checks first.
3. Use minimal, readable code with descriptive names and explicit steps. Include detailed Python docstrings, MATLAB help comments, and comments explaining purpose, units, assumptions, calculations, and edge cases. Avoid compressed expressions and unnecessary abstractions.
4. Where both languages implement a calculation, use shared documented synthetic examples, independently known expected results, and a justified numerical tolerance.
5. Report actual Python and MATLAB execution separately, provide reproduction commands, update the project record, and stop for student testing and understanding.

The next research microstep is to design a common, outcome-independent measurement/threshold-validation protocol. That protocol is proposed, not yet approved in detail. The implementation branch name and first coding microstep remain unselected.

## Repository status

This repository contains documentation and exclusions only. No rebuild implementation or thesis tests have run. MATLAB previously failed at startup; MATLAB execution and cross-language agreement remain unverified.

The initial setup commit was pushed to the private repository. Future commit/push policy remains unresolved. Local folder ownership and separate Codex project selection remain pending in the project record.

## Working rules

- Implement accepted research decisions one microstep at a time, then stop for review.
- Preserve original data and previous work. Keep participant data and credentials outside this repository.
- Use synthetic or approved de-identified examples for tests and label them clearly.
- Keep later outcome measurements separate from inputs available at the D0 decision point.
- Verify the exact staged files before every upload; ignore rules do not protect files already tracked by Git.

The authoritative project record is `PROJECT_STATE.md` in the parent `Rebuild_v3` folder, outside this repository. It contains the full measurement specifications, decision history, source limitations, and setup evidence. Planning prompts are also maintained outside the repository; they are not included on GitHub. This README summarizes state version 0.4.
