# TransitShortcut

**Is your exoplanet classifier detecting transits... or learning the dataset?**

TransitShortcut audits shortcuts in exoplanet classification. It compares a deliberately questionable metadata baseline against a baseline using only raw photometry, measures artifact prevalence, and refits models after removing feature groups. Its local workstation lets you inspect real light curves, prediction disagreements, and the evidence behind each audit.

It is scientific evaluation software. It does not discover or validate planets.

## Run locally

Requirements: Python 3.12+ (3.13 recommended), [uv](https://docs.astral.sh/uv/), Node.js 22.12+ with Corepack/pnpm, and an internet connection for initial dependency installation and data preparation. CPU only; no API keys or paid services.

```sh
uv sync --locked --python 3.13
corepack pnpm --dir web install --frozen-lockfile
corepack pnpm --dir web build

# Standard sample: 2,400 / 500 / 500, preserving original split membership
uv run transitshortcut data prepare --mode sample --seed 42
uv run transitshortcut data inspect
uv run transitshortcut train --experiment multiclass
uv run transitshortcut train --experiment cross-mission
uv run transitshortcut serve
```

Open [the workstation](http://127.0.0.1:8000). FastAPI serves the production frontend and [API documentation](http://127.0.0.1:8000/docs). Without prepared data, the UI shows setup instructions and no invented results. Synthetic examples exist only in automated tests.

For a smaller first experiment (the cohort used in the initial verification):

```sh
uv run transitshortcut data prepare --seed 42 --train-size 90 --validation-size 30 --test-size 30 --max-flux-mb 300 --revision 927f66212ecf8acf8a460c6e7efadc6dc8d1b9c7
uv run transitshortcut train --experiment multiclass
uv run transitshortcut train --experiment cross-mission
```

This reads approximately 132 MiB of compressed flux row-group data, plus metadata, Parquet footers, and range-read overhead. It saves only the 150 selected observations. Small mission subsets are explicitly warned about, and unreliable mission chart bars are withheld. See [measured findings](docs/findings.md).

For frontend development, run the backend, then `corepack pnpm --dir web dev`. Vite proxies `/api` to port 8000. To refresh data or models from the UI, complete CLI work and click **Refresh artifacts**. Jobs run synchronously through the CLI; the web server does not launch training jobs.

## Dataset and sampling

Source: [bingbangboom/exoplanet-transit-detection](https://huggingface.co/datasets/bingbangboom/exoplanet-transit-detection), by **bingbangboom**, under **CC BY 4.0**. Published counts are 23,567 observations across Kepler, K2, and TESS, with PLANET, FALSE_POSITIVE, and NO_SIGNAL labels. Actual snapshot counts are measured, not assumed.

The dataset publishes `train.parquet`, `val.parquet`, and `test.parquet`. Validation is normalized to `validation` locally. Every binary light curve is decoded as little-endian float32 after validating its byte length: raw 20,000, global 201, local 81, odd/even 201 each. All-nonfinite arrays fail validation; partial nonfinite raw arrays are median-imputed per example and counted. Exact zero detection never treats missing values as zero.

Remote preparation uses the supported `HfFileSystem` integration with PyArrow byte-range access. Metadata discovery reads only star ID, mission, and label across the original splits. It detects duplicate identifiers throughout those source splits before accepting a prepared snapshot.

**Sampling is deterministic, stratified within seeded row-group clusters.** Pure row sampling would touch almost every 1,000-row flux block even for a tiny cohort. The sampler determines a proportional mission/class allocation from all split metadata (with at least one per stratum when the budget permits), selects seeded row groups until they can supply that allocation, then samples within those groups. The manifest records candidate groups and selected source-row offsets. This preserves mission/class allocation as closely as rounding permits, but is **not a uniform random sample within each full-source stratum**. Row-group ordering can bias the cohort. Do not extrapolate results to the full dataset.

Sample mode caps compressed flux row-group reads at **512 MiB** by default. Metadata and small range-cache overhead are additional. If a changed layout, seed, or sample size needs more, preparation stops before reading the block that would exceed the budget. It never falls back to a full download. Increase `--max-flux-mb` deliberately or choose a smaller sample. Groups are processed one at a time. Full processing is explicit:

```sh
uv run transitshortcut data prepare --mode full --revision COMMIT_HASH
```

Full mode processes all original splits and may transfer approximately 1.95 GB. It is optional and was not executed during initial verification.

Prepared snapshots and extracted features are cached under `data/<dataset-id>/`. Model pipelines and immutable run results live under `artifacts/<run-id>/`. Both are ignored by Git. Set `TRANSITSHORTCUT_HOME` to change the storage root. Repeating preparation with the same pinned revision and configuration reuses the validated snapshot. Training, auditing, report export, and browsing work offline after preparation; preparation itself resolves the remote revision and needs network access.

Each manifest includes repository, resolved commit, UTC timestamp, seed, requested/actual split sizes, inspected metadata counts, mission/class cross-tabulation, artifact frequencies, selected identifiers, configuration, duplicate/nonfinite checks, sampling caveats, and snapshot SHA256. A changed configuration creates a separate dataset identity. The API only exposes completed experiments matching the active dataset.

## Models and experiments

Both models use scikit-learn Logistic Regression with fixed `C=1`, maximum 2,000 iterations, and the manifest seed. Numeric median imputation/scaling and categorical encoding are fitted on **training only**. Validation is evaluated separately. No hyperparameters are selected from test results, and no search is performed. Test predictions, metrics, ablations, plots, and exports share persisted artifacts.

| Model          | Inputs                                                                                                                                                    | Restrictions                                                                                                 |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Shortcut Model | Mission/cadence; transit-parameter availability and missingness; folded-array zero indicators/fractions; catalog-flag missingness                         | No identifiers, disposition, label source, confidence score, flag values, or photometric signal measurements |
| Signal Model   | Raw-flux variability, MAD/quantiles, negative excursions, asymmetry, local dip fractions/widths/contrast, window minima, fixed sample-lag autocorrelation | No mission, transit parameters, folded views, catalog flags, or provenance                                   |

Exact feature allowlists appear in the UI and saved reports. Signal features are deliberately modest proxies for photometric structure, not a period search. Near-constant flux is handled with a scale floor; robust standardized deviations are clipped to limit outlier effects. No physical durations or orbital periods are inferred from sample indices.

- **A — Multiclass:** train both models on the same PLANET/FALSE_POSITIVE/NO_SIGNAL cohort. Report overall and mission subset metrics, including Kepler. Missing true classes have null recall/F1, rather than invented zero recall.
- **B — Binary mission comparison:** train fresh pipelines on the same PLANET/FALSE_POSITIVE rows, then evaluate Kepler, K2, and TESS separately. This is a within-dataset breakdown, **not** training on one mission and testing on a held-out mission.
- **C — Refit ablations:** remove mission/cadence, transit availability, folded-zero features, or flag missingness from the Shortcut Model, one group at a time; refit on unchanged training rows; compare test macro F1, changed labels, and maximum absolute per-class score differences. Rank by baseline-minus-ablated F1. Negative drops are allowed.
- **D — Individual sensitivity:** compare saved baseline and ablated predictions for an evaluated target. Display actual features, all class probability estimates, and score differences. No synthetic metadata overrides are presented as observations.

Metrics include accuracy, balanced accuracy, macro/weighted F1, per-class precision/recall/F1/support, and confusion matrices with stable task ordering. For subsets with missing true classes, macro F1 and balanced accuracy average over supported true classes; absent-class entries remain null. Small-sample warnings appear if any task class has fewer than five observations. Mission chart values are withheld under that rule; raw diagnostic metrics remain in reports with warnings.

## Workstation

- **Overview:** observed composition, artifact counts, active revision, and completed runs.
- **Model Comparison:** experiment/run selection, filtered metrics and confusion matrices, binary mission comparisons, and error overlap counts. Mission charts are explicitly labeled as the unfiltered full test cohort. Other result panels update together.
- **Signal Explorer:** actual prepared observations, evaluated disagreements, five on-demand photometric views, min/max-envelope downsampling, target provenance, and ablation sensitivity. Train/validation targets can be browsed but have no final test prediction record.
- **Shortcut Audit:** class/mission artifact heatmap, refit comparisons, changed predictions by class/mission, largest score changes, and exports.

Raw curves use **sample index**. Folded views use **folded bin index / relative position**, without invented physical calibration. All-zero folded arrays show an unavailable annotation instead of a normal flat stellar curve. Zoom, pan, reset, and crosshair are available in the Plotly toolbar. Raw arrays load only for the selected target; list responses contain no flux arrays.

## Audit exports

```sh
uv run transitshortcut audit --experiment multiclass
uv run transitshortcut report --experiment multiclass --format markdown --output audit.md
uv run transitshortcut report --experiment multiclass --format json --output audit.json
uv run transitshortcut report --experiment multiclass --format csv --output predictions.csv
```

`--experiment` accepts a task name (latest completed matching run) or an exact run ID. Reports contain run time, Git version and source hash, dataset manifest, configurations, metrics, feature groups, ablations, limitations, and reproduction commands. Markdown includes the complete saved JSON evidence. CSV contains row-level baseline/ablation labels and class scores. UI download links use the same artifacts. Checksum failures are surfaced rather than silently mixing results.

## Interpretation and limits

A suspicious **correlation** is an observed label association. **Model sensitivity** means this pipeline changes after a group is removed and refitted. **Scientific generalization** requires suitable independent observations and is not established by either measurement.

NO_SIGNAL is Kepler-only in the published dataset. It has real raw photometry, but missing transit parameters and zero-filled folded arrays. Zero-filled folded views also occur for planet/false-positive entries with incomplete catalogs. Catalog coverage and labels depend on external classification processes. Some stellar/transit properties are unavailable. Construction choices, interpolation, and resampling can bias performance. Even the raw-only model can exploit photometric preprocessing differences; its feature restriction does not prove physical understanding. Scores are not calibrated confidence. Refit ablations can retain correlated substitute shortcuts and are not causal interventions. No bootstrap uncertainty estimates or independent validation set outside the published dataset are supplied.

## Verification and development

```sh
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
corepack pnpm --dir web test
corepack pnpm --dir web build
```

Tests use clearly identified, isolated synthetic fixtures. They cover decoding, malformed/nonfinite arrays, deterministic allocation, leakage checks, allowed features, raw-only edge cases, train-only encoders, pipeline persistence, absent classes, positive/negative shortcut controls, API filtering/pagination/exports, extrema-preserving downsampling, frontend loading/setup states, page navigation, target selection, and unavailable folded views. GitHub Actions runs the Python and frontend checks without downloading scientific data. Hosted CI execution is not implied by local checks.

The backend lives in `src/transitshortcut/`: `data.py`, `features.py`, `experiments.py`, `api.py`, and `cli.py`. The interface is in `web/src/`. A small number of cohesive modules keep the experimental path inspectable.

## License and attribution

Original code: [MIT](LICENSE). The dataset retains its separate [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/); this repository does not relicense it or include source observations in Git. Derived snapshots are sampled and normalized locally; features and model results are transformations of that source.

Credit the dataset creator **bingbangboom**, the [dataset card and citation](https://huggingface.co/datasets/bingbangboom/exoplanet-transit-detection/blob/main/README.md), NASA Kepler/K2/TESS, [MAST](https://archive.stsci.edu/), the [NASA Exoplanet Archive](https://doi.org/10.26133/NEA4), and [Lightkurve](https://ascl.net/1812.013). The source card also credits [Thompson et al. (2018), Kepler Robovetter](https://doi.org/10.3847/1538-4365/aab4f9), [Shallue & Vanderburg (2018), folded-view methodology](https://doi.org/10.3847/1538-3881/aa9e09), [Ricker et al. (2015), TESS](https://doi.org/10.1117/1.JATIS.1.1.014003), and [Howell et al. (2014), K2](https://doi.org/10.1086/676406).

```bibtex
@dataset{exoplanet_transit_detection_dataset,
  author = {bingbangboom},
  title = {Exoplanet Transit Detection Dataset},
  year = {2026},
  url = {https://huggingface.co/datasets/bingbangboom/exoplanet-transit-detection},
  note = {Multi-mission dataset (Kepler, K2, TESS) with NO_SIGNAL class},
  license = {CC-BY-4.0}
}
```
