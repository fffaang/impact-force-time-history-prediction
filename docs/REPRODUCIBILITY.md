# Reproduction guide

## 1. Inspect the dataset

Use Python 3.12 and `requirements-inference.txt`. Run `python scripts/verify_dataset.py`.
The check compares CSV ordinates against NPZ values exactly, recomputes descriptors,
checks 199/198 counts, and verifies all fixed split and known specimen-group constraints.
This is a computational integrity check, not new pointwise verification against every paper.

## 2. Reproduce saved predictions without training

The primary model (20261102) needs only the CPU inference requirements:

```shell
python scripts/replay_predictions.py --split 20261102 --output output/replay
```

The other two predetermined splits selected **stored neural decoders**, not the primary
PCA8-ExtraTrees production branch. Install `requirements-replay.txt` to replay all three
on CPU, including their original PyTorch classes and CatBoost components:

```shell
python -m pip install -r requirements-replay.txt
python scripts/replay_predictions.py --split all --output output/replay
```

The replay checks all 198 eligible predictions per partition, the 40 held-out histories,
training IDs, raw-input-only inference, batch invariance, nonnegative force, four-output
consistency, and exact piecewise-linear physical-time scores against the stored CSVs.
Tolerance is rtol 1e-9 / atol 1e-8 for descriptors and force ordinates. CPU/library
portability must be checked rather than inferred from a successful import.

## 3. Tables, SHAP, and source-block uncertainty

`python scripts/export_workbooks.py --output output/workbooks` creates editable Excel
tables from the frozen CSVs with openpyxl (`requirements-tools.txt`). Source values,
not document rendering, are the reproducible scientific deliverable.

`python scripts/explain_predictions.py --check --output output/shap` recomputes the
primary five-output SHAP function and compares with stored values. It uses 24 training
background records, 32 random permutations plus their reverses, seed 20261103,
raw-input coalitions, recomputed derived features, and log/logit output scales.
The derived Astar is explained through the real calculation, not a different regressor.
This may take several minutes. Background-based explanations are not causal effects.

`src/metrics.py` contains the physical-time metric and 2,000-replicate lineage/source-block
paired-bootstrap implementation. Its stored comparison CSVs retain negative results and
invalid/degenerate-resample counts. Three split results must remain separate.

## 4. Refit or repeat the complete search in a fresh workspace

The original CUDA search used Python 3.12, PyTorch 2.7.1+cu128, scikit-learn 1.5.2,
the versions in `requirements-training.txt`, and one RTX 4090 GPU. The original
environment inventories are included as version records, not universal platform locks.
Install training dependencies in a separate environment with a compatible NVIDIA driver.
The CPU prediction environment does not require CUDA.

First run scientific tests and a short **training-only** smoke reproduction:

```shell
python -m pip install -r requirements-training.txt
python scripts/reproduce_training.py --workdir ../training-smoke --device cuda --smoke
```

Explicitly repeat the formal experiment:

```shell
python scripts/reproduce_training.py --workdir ../training-reproduction --device cuda --run
```

The destination must be empty and outside the repository. The wrapper copies scientific
source, frozen numerical inputs, known-group split indices and configuration, runs tests
and an audit, and records a **new portable-source freeze** before fitting. It does not copy
published results, fitted models, old caches, or the original computational seal into the
new run. The original documented 30-hour search / 48-hour total budgets and training-only
three-fold model selection are retained. Budget exhaustion is reported, not relabelled as
successful completion. `--device cpu` is an execution option, not a promise of identical
CUDA training behavior or completion within the same budget.

Low-level stages retain the original interface inside that prepared workspace:
`python scripts/run_experiment.py --config config/split811_20261102.yml --stage all --split all`.
Do not run fitting stages inside the published repository. Fixed indices and seeds are
not replaced based on holdout scores. Four-anchor target consistency, baseline/optimized
comparisons, independent refits, group isolation, and negative scores are tested.

The saved `control_catalog.joblib` supplies historical **unfitted architecture settings**;
no old fitted coefficients, preprocessing or weights are reused for A/B retraining. The
original retained-model bundle is unnecessary when the catalog is present. Legacy source
classes remain because A/B model deserialization and refitting require them.

## 5. Releases, hashes, and historical evidence

The repository includes all three C models. Each partition's A/B models and independent
refits are distributed in a separate Release ZIP; identical model files are represented
once with explicit alias mappings. Extract to the repository root only if those controls
are needed. `scripts/restore_control_aliases.py` recreates aliases after verifying hashes.

`SHA256SUMS.csv` describes this public release, excluding generated output and Git metadata.
`provenance/original_freeze.json` and `original_selection_seal.json` describe the original
completed experiment. They contain historical paths and hashes for original run files
that were intentionally not all republished. They are not a claim that the portable
release source has the identical source hash. Data NPZ/metadata and final model binaries
are byte-identical; adapted entry points, path normalization, documentation, and new CSV
views have separate release checksums. `provenance/copied_files.json` records original
copy hashes; current values are always checked against the release manifest.

Full search is not rerun merely to publish this snapshot. Release QA distinguishes
new frozen-inference checks and short CPU compatibility tests from the historical
full CUDA training and refit evidence.
