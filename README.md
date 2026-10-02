# Adaptive VR Difficulty Recommendation

[![Python tests](https://github.com/MKhedr133/Thesis-v3/actions/workflows/python-tests.yml/badge.svg)](https://github.com/MKhedr133/Thesis-v3/actions/workflows/python-tests.yml)

MSc Robotics thesis project at TU Delft investigating how performance,
behaviour, and perceived task difficulty can be used to support adaptive
difficulty selection in a VR rehabilitation task.

The project combines VR data processing, statistical modelling, predictive
analysis, automated testing, and adaptive decision support.

---

## Project overview

A rehabilitation participant completes a shopping task inside a virtual
supermarket.

The task can be made more difficult in three different ways:

- **Visual:** increasing visual activity and distractions
- **Auditory:** increasing environmental sound
- **Cognitive:** increasing the difficulty of a concurrent mental task

The objective is to use measurements collected during the task to determine
how the participant responds to difficulty and support a therapist when
selecting the next difficulty level.

### Experimental scale

| Item | Value |
| --- | ---: |
| Participants | 32 |
| Conditions | 3 |
| Difficulty levels | 4 |
| Trials per participant | 12 |
| Total trials | 384 |

Difficulty levels are `D0`, `D2`, `D6`, and `D10`.

---

## System workflow

```mermaid
flowchart LR
    A[VR supermarket trials] --> B[Raw behavioural data]
    B --> C[Python feature extraction]

    C --> D[Performance measures]
    C --> E[Behavioural measures]
    C --> F[Perceived difficulty]

    D --> G[Statistical analysis]
    E --> G
    F --> G

    G --> H[Difficulty modelling]
    H --> I[Adaptive difficulty decision support]
    I --> J[Therapist decision]
