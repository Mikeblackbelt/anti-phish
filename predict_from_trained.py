"""
predict.py — Load a trained phishing classifier and use it to make predictions.

Usage:
  1) Predict on a CSV of new emails (same feature columns as training data):
       python predict_from_trained.py --model phishing_model.joblib --data new_emails.csv --out predictions.csv

  2) Interactively test a single email from the command line:
       python predict_from_trained.py --model phishing_model.joblib --interactive

  3) Quick built-in demo using a few example rows shipped in this file:
       python predict_from_trained.py --model phishing_model.joblib --demo
"""

import argparse
import sys

import joblib
import pandas as pd


def load_model(path: str):
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["label_col"], bundle["text_col"], bundle["feature_cols"]


def predict_dataframe(pipeline, feature_cols, df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = None  # let the pipeline's imputers handle missing columns
    X = df[feature_cols]
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]
    out = df.copy()
    out["predicted_label"] = ["phishing" if p == 1 else "not_phishing" for p in preds]
    out["phishing_probability"] = probs
    return out


def run_interactive(pipeline, feature_cols, text_col):
    print("Enter values for each feature (leave blank to skip / use default).")
    row = {}
    for col in feature_cols:
        if col == text_col:
            prompt = f"{col} (email text): "
        else:
            prompt = f"{col}: "
        val = input(prompt)
        if val.strip() == "":
            row[col] = None
        else:
            row[col] = val
    df = pd.DataFrame([row])
    # Try to coerce numeric-looking values back to numbers.
    for col in df.columns:
        if col != text_col:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    result = predict_dataframe(pipeline, feature_cols, df)
    label = result.loc[0, "predicted_label"]
    prob = result.loc[0, "phishing_probability"]
    print(f"\nPrediction: {label.upper()}  (phishing probability: {prob:.4f})")


def run_demo(pipeline, feature_cols, text_col):
    # Example rows shaped like the ethancratchley "Email Phishing Dataset" features:
    # num_words, num_unique_words, num_stopwords, num_links, num_unique_domains,
    # num_email_addresses, num_spelling_errors, num_urgent_keywords
    examples = [
        {
            # short, link-heavy, urgency-heavy, several distinct suspicious domains
            "num_words": 45,
            "num_unique_words": 30,
            "num_stopwords": 12,
            "num_links": 6,
            "num_unique_domains": 5,
            "num_email_addresses": 1,
            "num_spelling_errors": 4,
            "num_urgent_keywords": 7,
        },
        {
            # longer, well-formed, no links/urgency — looks like a normal work email
            "num_words": 180,
            "num_unique_words": 95,
            "num_stopwords": 60,
            "num_links": 0,
            "num_unique_domains": 0,
            "num_email_addresses": 1,
            "num_spelling_errors": 1,
            "num_urgent_keywords": 0,
        },
    ]
    df = pd.DataFrame(examples)
    result = predict_dataframe(pipeline, feature_cols, df)
    for i, row in result.iterrows():
        print(f"\nExample {i + 1}: {dict(row[feature_cols])}")
        print(f"  Prediction: {row['predicted_label'].upper()} (probability: {row['phishing_probability']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Use a trained phishing classifier.")
    parser.add_argument("--model", required=True, help="Path to the trained model (.joblib) from train.py")
    parser.add_argument("--data", default=None, help="CSV of new emails to classify.")
    parser.add_argument("--out", default="predictions.csv", help="Where to write predictions (used with --data).")
    parser.add_argument("--interactive", action="store_true", help="Interactively enter one email to classify.")
    parser.add_argument("--demo", action="store_true", help="Run a quick demo with two built-in example emails.")
    args = parser.parse_args()

    pipeline, label_col, text_col, feature_cols = load_model(args.model)
    print(f"Loaded model. Expected feature columns: {feature_cols}")

    if args.demo:
        run_demo(pipeline, feature_cols, text_col)
    elif args.interactive:
        run_interactive(pipeline, feature_cols, text_col)
    elif args.data:
        df = pd.read_csv(args.data)
        result = predict_dataframe(pipeline, feature_cols, df)
        result.to_csv(args.out, index=False)
        n_phish = (result["predicted_label"] == "phishing").sum()
        print(f"Classified {len(result)} emails: {n_phish} predicted phishing, {len(result) - n_phish} not phishing.")
        print(f"Saved predictions to {args.out}")
    else:
        print("Nothing to do — pass --data, --interactive, or --demo. See --help.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())