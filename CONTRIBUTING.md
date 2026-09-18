# Contributing workflow

This repository uses a lightweight pull request workflow for thesis software development.

## 1. Start from an issue

Create or select a GitHub issue that defines:

- the problem or objective;
- the intended scope;
- acceptance criteria;
- any research or data boundary that must not be crossed.

The issue should define what "done" means before implementation starts.

## 2. Create a focused branch

Create the branch from the latest `main`.

Recommended prefixes:

- `feature/` for new functionality;
- `fix/` for corrections;
- `refactor/` for structural changes without intended behaviour changes;
- `test/` for test-only work;
- `docs/` for documentation;
- `chore/` for repository or tooling work.

Keep one main purpose per branch.

## 3. Make small, meaningful commits

Use short imperative commit messages such as:

- `feat: add difficulty prediction baseline`
- `fix: use raw TMT-B seconds in candidate models`
- `test: verify raw TMT-B formulas`
- `docs: update coefficient interpretation`

A commit should represent one logical change that could be understood or reverted independently.

## 4. Push and open a pull request

Open a pull request into `main` and link the relevant issue.

The pull request must distinguish four states:

1. code implemented;
2. automated tests passed;
3. official local analysis executed;
4. scientific result verified.

These are not equivalent.

## 5. Automated checks

GitHub Actions runs the repository's synthetic Python unit tests on pull requests into `main`.

Passing CI means the committed code passed those automated checks in the GitHub environment. It does not mean participant data were analysed, the official thesis analysis was executed, or a scientific result was verified.

Official Python and MATLAB analysis remains in the local thesis environment unless explicitly changed.

## 6. Self-review before merge

Review the pull request's **Files changed** tab as if reviewing another developer's work.

Check:

- the issue is actually solved;
- unrelated files were not changed;
- names and comments are understandable;
- tests cover the important behaviour;
- participant data, generated outputs, secrets, and local files were not committed;
- documentation or `PROJECT_STATE.md` needs updating.

## 7. Merge

Merge only after the intended checks and review are complete.

For small focused pull requests, prefer **Squash and merge** so that `main` keeps one clear commit per completed task while the pull request retains the detailed development history.

Delete the branch after a successful merge.

## 8. Thesis state recording

GitHub issues and pull requests record technical development history.

`PROJECT_STATE.md` records the current accepted methodological and project state. Update it only when a meaningful accepted decision, verified implementation result, verified execution result, unresolved question, or next task changes.
