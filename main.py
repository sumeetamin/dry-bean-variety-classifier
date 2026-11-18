"""Reproduce a stratified UCI Dry Bean classification benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo


DATASET_ID = 602
PROJECT_TITLE = "Dry Bean Variety Classifier"
RANDOM_SEED = 42
TEST_SIZE = 0.25


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load only UCI Dry Bean; fail if the source schema changes unexpectedly."""
    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if targets.shape[1] != 1:
        raise ValueError(f"Expected one class target, found {targets.shape[1]}.")
    if features.empty or targets.isna().any(axis=1).all():
        raise ValueError("UCI returned no usable Dry Bean records.")
    if not all(pd.api.types.is_numeric_dtype(features[column]) for column in features):
        raise ValueError("Dry Bean feature schema must remain numeric.")
    keep = targets.iloc[:, 0].notna()
    return features.loc[keep].reset_index(drop=True), targets.loc[keep].iloc[:, 0].reset_index(drop=True)


def build_model(features: pd.DataFrame) -> Pipeline:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=PROJECT_TITLE)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    features, labels = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    model = build_model(x_train)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    class_order = model.named_steps["classifier"].classes_

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)

    metrics = {
        "project": PROJECT_TITLE,
        "dataset_id": DATASET_ID,
        "dataset": "UCI Dry Bean",
        "rows": int(len(features)),
        "features": list(features.columns),
        "target": str(labels.name or "Class"),
        "classes": [str(label) for label in class_order],
        "split": "stratified random 75/25 holdout; seed=42",
        "task": "classification",
        "holdout": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
            "log_loss": float(log_loss(y_test, probabilities, labels=class_order)),
            "macro_ovr_roc_auc": float(roc_auc_score(
                y_test, probabilities, labels=class_order, multi_class="ovr", average="macro",
            )),
            "majority_baseline_accuracy": float(accuracy_score(y_test, baseline_predictions)),
            "per_class": classification_report(
                y_test, predictions, output_dict=True, zero_division=0,
            ),
        },
        "limitations": (
            "Public UCI tabular benchmark with image-derived measurements. "
            "A random holdout does not test new farms, cameras, harvest conditions, or deployment drift."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    holdout = pd.DataFrame({
        "actual": y_test.astype(str).to_numpy(),
        "predicted": predictions.astype(str),
        "max_class_probability": probabilities.max(axis=1),
    })
    holdout.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, args.output_dir / "model.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
