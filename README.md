# Dry Bean Variety Classifier

Classify seven dry-bean varieties using measured geometric and shape features.

## Dataset

[UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry%2Bbean) (dataset ID 602), 13,611 records. The source is public and is fetched by the runner; the raw dataset is not committed.

## Run

```bash
python -m pip install -r requirements.txt
python main.py --output-dir artifacts
```

The runner saves metrics, holdout predictions, and a reusable model in the ignored output directory. The public [`reports/benchmark.json`](reports/benchmark.json) contains aggregate results only; no row-level records are published.

## Holdout results

A stratified random 75/25 split (seed 42) is used. Numeric inputs are median-imputed; the random-forest classifier uses 400 trees and balanced subsampling.

| Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Log loss | Macro OVR ROC-AUC | Majority baseline accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.920 | 0.932 | 0.933 | 0.920 | 0.247 | 0.994 | 0.261 |

Log loss and ROC-AUC describe probabilities, but no separate calibration dataset or calibration correction is used. Full aggregate metrics and the exact split seed are in [`reports/benchmark.json`](reports/benchmark.json).

## Limits

This benchmark covers image-derived measurements from a limited set of varieties and sources; the random split does not test new farms, cameras, or harvest conditions. These are one-time benchmark results, not service-level, production, or causal claims.

## Reproducibility

The model and split use seed 42. Re-running downloads source data from UCI and regenerates local artifacts. Model binaries, raw inputs, and row-level predictions are excluded from this public repository.
