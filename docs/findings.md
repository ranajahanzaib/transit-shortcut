# Initial measured findings

These are actual small-cohort experiments, not bundled UI benchmark values. The application always reads local completed artifacts. This document records the initial verification; new runs may differ.

Source revision: `927f66212ecf8acf8a460c6e7efadc6dc8d1b9c7`. Seed: 42. Cohort: 90 train / 30 validation / 30 test. Snapshot SHA256: `1fa34069a58472838f7d5ac67ddb0aee019a79e3588fbb4679c996e3ccbc73b3`.

All 23,567 source metadata rows were inspected; 150 observations selected. Candidate row groups: `{'train': [14], 'validation': [2], 'test': [2]}`. Compressed flux row-group data: 131.89 MiB, plus metadata and range-read overhead. Sampling used seeded row-group clusters with mission/class allocation from whole-split metadata. No duplicate source IDs were detected.

## Observed construction traces

The 150 selected rows contained 61 PLANET, 39 FALSE_POSITIVE, and 50 NO_SIGNAL examples. All 50 NO_SIGNAL examples were Kepler observations. All had missing transit parameters and zero folded views. Four additional planet/false-positive examples had zero folded views: the artifact was not exclusive to NO_SIGNAL. These are cohort observations, not a remeasurement of the full dataset’s folded-array frequencies.

## Evaluation

| Task          | Model    | Test n | Accuracy | Balanced accuracy | Macro F1 |
| ------------- | -------- | -----: | -------: | ----------------: | -------: |
| multiclass    | shortcut |     30 |   0.8333 |            0.8472 |   0.8329 |
| multiclass    | signal   |     30 |   0.5667 |            0.5333 |   0.5333 |
| cross-mission | shortcut |     20 |   0.8000 |            0.8333 |   0.8000 |
| cross-mission | signal   |     20 |   0.5500 |            0.5000 |   0.4872 |

## Refit ablations

Positive F1 drop means the ablated pipeline scored lower. Negative drops are improvements after removal. This is model-specific refit sensitivity, not causal attribution.

| Task          | Removed group        | Macro F1 drop | Changed predictions | Mean maximum score change |
| ------------- | -------------------- | ------------: | ------------------: | ------------------------: |
| multiclass    | Transit availability |        0.0349 |                1/30 |                    0.0694 |
| multiclass    | Folded zeros         |        0.0000 |                0/30 |                    0.0035 |
| multiclass    | Mission/cadence      |       -0.0338 |                1/30 |                    0.0160 |
| multiclass    | Flag missingness     |       -0.0338 |                1/30 |                    0.0174 |
| cross-mission | Transit availability |        0.1030 |                2/20 |                    0.0952 |
| cross-mission | Mission/cadence      |        0.0000 |                0/20 |                    0.0166 |
| cross-mission | Folded zeros         |        0.0000 |                0/20 |                    0.0019 |
| cross-mission | Flag missingness     |        0.0000 |                0/20 |                    0.0401 |

Removing transit availability reduced test macro F1 in both tasks. Removing folded-zero indicators did not change test labels in either task, despite strong zero/label correlation. Other remaining availability indicators can substitute for the removed group; lack of a drop does not establish that the artifact is harmless.

Mission-specific binary supports were Kepler 10 (4 planet / 6 false positive), K2 3 (2 / 1), and TESS 7 (6 / 1). Every mission had a class with fewer than five test observations. The UI withholds their comparative bars. No claim of reliable cross-mission generalization is justified.

## Reproduction and provenance

Use the small pinned-cohort command in README, then train both tasks. No configuration was selected using test performance. Validation was scored separately. Published splits were retained, and preprocessing was fitted only on training observations. Test n=30 is too small for a stable benchmark; sampling has an additional cluster-selection limitation.

- `multiclass-e56c4cf982` completed `2026-11-02T12:00:43.186774+00:00`. Working source SHA256: `4a29ff40ddfe441a186d01eaf13b5a8724f91029a6fb6429b280f40b31001fb8`; Git base: `38f727690ee5c695724aa68b340ca48e0754a94a`.
- `cross-mission-93acffa51a` completed `2026-11-02T12:00:44.947757+00:00`. Working source SHA256: `4a29ff40ddfe441a186d01eaf13b5a8724f91029a6fb6429b280f40b31001fb8`; Git base: `38f727690ee5c695724aa68b340ca48e0754a94a`.

Full-data execution, independent observations, held-out-mission training, probability calibration, and hosted CI were not validated. No planet discovery or physical interpretation is established.
