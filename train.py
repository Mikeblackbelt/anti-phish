"""
train.py — High-recall phishing classifier on the Kaggle engineered-feature dataset.
Dataset: "Email Phishing Dataset" (Kaggle, ethancratchley)
https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset
Features available (no raw text):
  num_words, num_unique_words, num_stopwords, num_links,
  num_unique_domains, num_email_addresses, num_spelling_errors,
  num_urgent_keywords, label
Improvements over the baseline:
  - Rich ratio / interaction features derived only from the columns above
  - XGBoost with scale_pos_weight + early stopping on PR-AUC
  - Decision threshold tuned for maximum recall at a precision floor
  - PR-AUC and ROC-AUC reported
Usage:
    python train.py --data path/to/dataset.csv --out model.joblib
"""
import argparse
import glob
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

KAGGLE_DATASET_SLUG = "ethancratchley/email-phishing-dataset"

LABEL_CANDIDATES = [
    "label", "is_phishing", "phishing", "target", "class",
    "email_type", "type", "y",
]

# Columns known to exist in this dataset (used for feature engineering)
RAW_FEATURE_COLS = [
    "num_words",
    "num_unique_words",
    "num_stopwords",
    "num_links",
    "num_unique_domains",
    "num_email_addresses",
    "num_spelling_errors",
    "num_urgent_keywords",
]


def download_via_kagglehub(slug: str = KAGGLE_DATASET_SLUG) -> str:
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Install it with:\n"
            "  pip install kagglehub\n"
            "then re-run this script."
        )
    print(f"Downloading '{slug}' via kagglehub (this may take a while)...")
    try:
        dataset_dir = kagglehub.dataset_download(slug)
    except Exception as e:
        raise SystemExit(
            "Failed to download dataset via kagglehub.\n"
            f"Error: {e}\n\n"
            "Make sure you have Kaggle API credentials set up:\n"
            " 1. Go to https://www.kaggle.com/settings -> API -> 'Create New Token'\n"
            " 2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json\n"
            " (or set KAGGLE_USERNAME and KAGGLE_KEY environment variables)\n"
        )
    print(f"Dataset downloaded to: {dataset_dir}")
    csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise SystemExit(f"No CSV file found in downloaded dataset directory: {dataset_dir}")
    if len(csv_files) > 1:
        print(f"Multiple CSV files found, using the first one: {csv_files[0]}")
    return csv_files[0]


def find_label_column(df: pd.DataFrame) -> str:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in LABEL_CANDIDATES:
        if cand in lower_cols:
            return lower_cols[cand]
    for c in reversed(df.columns):
        vals = set(pd.unique(df[c].dropna()))
        if vals <= {0, 1} or vals <= {0.0, 1.0}:
            return c
    raise ValueError(
        "Could not auto-detect the label column. "
        f"Columns available: {list(df.columns)}. "
        "Pass --label-col explicitly."
    )


def normalize_label(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        mapping = {
            "phishing": 1, "phish": 1, "spam": 1, "malicious": 1, "1": 1, "yes": 1, "true": 1,
            "legitimate": 0, "safe": 0, "ham": 0, "benign": 0, "not phishing": 0,
            "0": 0, "no": 0, "false": 0,
        }
        series = series.astype(str).str.strip().str.lower().map(mapping).fillna(series)
    return series.astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a richer feature matrix from the 8 engineered columns only.
    All operations are row-wise and leakage-free.
    """
    out = df.copy()

    # Safe denominators
    words = out["num_words"].clip(lower=1).astype(float)
    links = out["num_links"].clip(lower=0).astype(float)
    links_safe = links.replace(0, np.nan)  # for ratios that should be NaN when no links

    # --- Density / ratio features (most informative for phishing) ---
    out["unique_word_ratio"] = out["num_unique_words"] / words
    out["stopword_ratio"] = out["num_stopwords"] / words
    out["spelling_error_rate"] = out["num_spelling_errors"] / words
    out["urgent_keyword_density"] = out["num_urgent_keywords"] / words
    out["links_per_word"] = out["num_links"] / words
    out["email_addrs_per_word"] = out["num_email_addresses"] / words

    # Domain diversity among links
    out["domains_per_link"] = out["num_unique_domains"] / links_safe
    out["domains_per_link"] = out["domains_per_link"].fillna(0.0)

    # Lexical richness proxies
    out["stopword_to_unique"] = out["num_stopwords"] / out["num_unique_words"].clip(lower=1)
    out["error_to_unique"] = out["num_spelling_errors"] / out["num_unique_words"].clip(lower=1)

    # --- Binary / count indicators ---
    out["has_links"] = (out["num_links"] > 0).astype(np.int8)
    out["has_multiple_links"] = (out["num_links"] >= 2).astype(np.int8)
    out["has_multiple_domains"] = (out["num_unique_domains"] >= 2).astype(np.int8)
    out["has_email_address"] = (out["num_email_addresses"] > 0).astype(np.int8)
    out["has_spelling_errors"] = (out["num_spelling_errors"] > 0).astype(np.int8)
    out["has_urgent_keywords"] = (out["num_urgent_keywords"] > 0).astype(np.int8)
    out["high_urgency"] = (out["num_urgent_keywords"] >= 2).astype(np.int8)

    # Short / long email flags (phishing often short + urgent)
    out["is_short"] = (out["num_words"] <= 50).astype(np.int8)
    out["is_very_short"] = (out["num_words"] <= 20).astype(np.int8)
    out["is_long"] = (out["num_words"] >= 300).astype(np.int8)

    # Interaction: urgency + links (classic phishing pattern)
    out["urgent_and_has_links"] = (
        out["has_urgent_keywords"] * out["has_links"]
    ).astype(np.int8)
    out["urgent_links_score"] = out["num_urgent_keywords"] * out["num_links"]
    out["errors_and_links"] = (
        out["has_spelling_errors"] * out["has_links"]
    ).astype(np.int8)

    # Log-scaled counts (helps tree splits on heavy-tailed counts)
    for col in RAW_FEATURE_COLS:
        out[f"log1p_{col}"] = np.log1p(out[col].clip(lower=0))

    return out


def get_feature_matrix(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Return X and the list of feature column names (everything except label)."""
    engineered = engineer_features(df)
    feature_cols = [c for c in engineered.columns if c != label_col]
    # Keep only numeric
    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(engineered[c])
    ]
    X = engineered[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, feature_cols


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    min_precision: float = 0.15,
) -> tuple[float, float, float, float]:
    """
    Maximise recall subject to precision >= min_precision.
    Returns (threshold, precision, recall, f1).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    precision = precision[:-1]
    recall = recall[:-1]

    mask = precision >= min_precision
    if mask.any():
        scores = np.where(mask, recall + 1e-6 * precision, -1.0)
        best_idx = int(np.argmax(scores))
    else:
        best_idx = int(np.argmax(recall))

    p = float(precision[best_idx])
    r = float(recall[best_idx])
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(thresholds[best_idx]), p, r, f1


def main():
    parser = argparse.ArgumentParser(
        description="High-recall phishing classifier (XGBoost + engineered ratios)."
    )
    parser.add_argument("--data", default=None, help="Path to dataset CSV.")
    parser.add_argument("--kaggle-slug", default=KAGGLE_DATASET_SLUG)
    parser.add_argument("--out", default="phishing_model.joblib")
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.15,
        help="Minimum precision when maximising recall (default: 0.15).",
    )
    args = parser.parse_args()

    data_path = args.data or download_via_kagglehub(args.kaggle_slug)
    print(f"Loading dataset from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")

    label_col = args.label_col or find_label_column(df)
    print(f"Using label column: '{label_col}'")
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = normalize_label(df[label_col])

    # Ensure expected raw columns exist (fill missing with 0)
    for col in RAW_FEATURE_COLS:
        if col not in df.columns:
            print(f"Warning: expected column '{col}' missing — filling with 0")
            df[col] = 0

    counts = df[label_col].value_counts().sort_index()
    n_neg, n_pos = int(counts.get(0, 0)), int(counts.get(1, 0))
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"Class distribution: not_phishing={n_neg}, phishing={n_pos} "
          f"(positive rate={n_pos / (n_neg + n_pos):.4f})")
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    X, feature_cols = get_feature_matrix(df, label_col)
    y = df[label_col]
    print(f"Feature count after engineering: {len(feature_cols)}")
    print(f"Features: {feature_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Small validation split from training data for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=args.seed, stratify=y_train
    )

    clf = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    print("Training XGBoost with early stopping on PR-AUC...")
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"Best iteration: {clf.best_iteration}")

    print("\nEvaluating on held-out test set...")
    probs = clf.predict_proba(X_test)[:, 1]

    preds_default = (probs >= 0.5).astype(int)

    best_thresh, best_p, best_r, best_f1 = find_best_threshold(
        y_test.to_numpy(), probs, min_precision=args.min_precision
    )
    preds_tuned = (probs >= best_thresh).astype(int)

    print(f"Best threshold (max recall @ precision>={args.min_precision}): "
          f"{best_thresh:.4f}  (P={best_p:.3f}, R={best_r:.3f}, F1={best_f1:.3f})")

    try:
        roc_auc = roc_auc_score(y_test, probs)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_test, probs)
    except ValueError:
        pr_auc = float("nan")

    print(f"\nROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")

    # Feature importance (top 15)
    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    print("\nTop 15 features by gain importance:")
    print(importances.sort_values(ascending=False).head(15).to_string())

    print(f"\n--- Default threshold (0.50) ---")
    print(f"Accuracy: {accuracy_score(y_test, preds_default):.4f}")
    print(classification_report(y_test, preds_default, target_names=["not_phishing", "phishing"]))
    print("Confusion matrix ([[TN, FP], [FN, TP]]):")
    print(confusion_matrix(y_test, preds_default))

    print(f"\n--- Recall-optimised threshold ({best_thresh:.4f}) ---")
    print(f"Accuracy: {accuracy_score(y_test, preds_tuned):.4f}")
    print(classification_report(y_test, preds_tuned, target_names=["not_phishing", "phishing"]))
    print("Confusion matrix ([[TN, FP], [FN, TP]]):")
    print(confusion_matrix(y_test, preds_tuned))

    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
            "label_col": label_col,
            "decision_threshold": best_thresh,
            "min_precision": args.min_precision,
            "scale_pos_weight": scale_pos_weight,
            "raw_feature_cols": RAW_FEATURE_COLS,
        },
        args.out,
    )
    print(f"\nSaved model (threshold={best_thresh:.4f}) to {args.out}")


if __name__ == "__main__":
    sys.exit(main())