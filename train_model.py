"""
train_model.py - Model Training and Evaluation Pipeline
Project: Student Announcement Intelligence System (SWYNEX Internship Task 2)
Author: Diya Patel

This module loads the announcement dataset, performs validation, splits into
stratified training and test sets, trains an NLP Pipeline (TF-IDF + Logistic Regression),
evaluates performance on unseen test data, and persists the fitted model and reports.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def get_project_paths():
    """Resolve and return absolute project paths."""
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "announcements.csv"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    model_file = models_dir / "announcement_classifier.joblib"
    metrics_file = reports_dir / "evaluation_metrics.json"
    report_file = reports_dir / "classification_report.txt"
    matrix_file = reports_dir / "confusion_matrix.png"
    classes_file = reports_dir / "classes.json"

    # Ensure output directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "data_path": data_path,
        "models_dir": models_dir,
        "reports_dir": reports_dir,
        "model_file": model_file,
        "metrics_file": metrics_file,
        "report_file": report_file,
        "matrix_file": matrix_file,
        "classes_file": classes_file
    }


def load_and_validate_data(csv_path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV and perform strict data validation.
    
    Checks:
    - File existence
    - Required columns ('announcement_text', 'category')
    - Removal of nulls, whitespace-only entries, and duplicates
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected path: {csv_path}. "
            "Please ensure data/announcements.csv exists."
        )

    print(f"[1/6] Loading dataset from: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = {"announcement_text", "category"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    initial_count = len(df)

    # Clean text and categories
    df["announcement_text"] = df["announcement_text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().str.upper()

    # Drop nulls or empty strings
    df = df[df["announcement_text"] != ""]
    df = df[df["category"] != ""]
    df = df.dropna(subset=["announcement_text", "category"])

    # Drop duplicate announcements if any
    df = df.drop_duplicates(subset=["announcement_text"]).reset_index(drop=True)

    cleaned_count = len(df)
    dropped = initial_count - cleaned_count
    if dropped > 0:
        print(f"      Removed {dropped} duplicate/invalid rows.")
    print(f"      Valid samples available: {cleaned_count}")

    # Check class distribution
    class_counts = df["category"].value_counts()
    print("      Class distribution:")
    for cat, count in class_counts.items():
        print(f"        - {cat:<12}: {count} samples")

    if len(class_counts) < 2:
        raise ValueError("Dataset must contain at least 2 categories for classification.")

    return df


def build_pipeline() -> Pipeline:
    """
    Construct a scikit-learn Pipeline with TF-IDF Vectorizer and Logistic Regression.
    
    Pipeline Steps:
    1. tfidf: TfidfVectorizer with sublinear TF scaling and unigram + bigram features.
    2. classifier: LogisticRegression with balanced class weights and fixed random state.
    """
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                    max_features=5000,
                    token_pattern=r"(?u)\b\w+\b"
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                    solver="lbfgs"
                )
            )
        ]
    )
    return pipeline


def train_and_evaluate():
    """Run end-to-end training, testing, reporting, and model artifact serialization."""
    paths = get_project_paths()
    print("=" * 70)
    print("  STUDENT ANNOUNCEMENT INTELLIGENCE SYSTEM — MODEL TRAINING")
    print("=" * 70)

    # 1. Load and validate
    df = load_and_validate_data(paths["data_path"])

    X = df["announcement_text"]
    y = df["category"]

    # 2. Stratified Train-Test Split (80% train, 20% test)
    print("\n[2/6] Splitting dataset (80% train, 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"      Training samples : {len(X_train)}")
    print(f"      Test samples     : {len(X_test)}")

    # 3. Model construction and fitting
    print("\n[3/6] Fitting scikit-learn Pipeline (TF-IDF + LogisticRegression)...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("      Model training complete.")

    # 4. Evaluation on unseen test data
    print("\n[4/6] Evaluating on held-out test data...")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report_text = classification_report(y_test, y_pred, digits=4)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "-" * 70)
    print(f"  Test Accuracy   : {accuracy * 100:.2f}%")
    print(f"  Macro F1-Score  : {macro_f1:.4f}")
    print(f"  Weighted F1-Score: {weighted_f1:.4f}")
    print("-" * 70)
    print("\nClassification Report:\n")
    print(report_text)

    # 5. Save artifacts & reports
    print("[5/6] Saving evaluation reports & plots...")
    labels = sorted(list(pipeline.classes_))

    # Save metrics JSON
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size": 0.20,
        "random_state": 42,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "classes": labels,
        "classification_report": report_dict
    }

    with open(paths["metrics_file"], "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"      Saved metrics JSON -> {paths['metrics_file'].name}")

    # Save classification report text
    with open(paths["report_file"], "w", encoding="utf-8") as f:
        f.write("STUDENT ANNOUNCEMENT INTELLIGENCE SYSTEM — CLASSIFICATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Test Accuracy    : {accuracy * 100:.2f}%\n")
        f.write(f"Macro F1-Score   : {macro_f1:.4f}\n")
        f.write(f"Weighted F1-Score: {weighted_f1:.4f}\n\n")
        f.write("Detailed Breakdown:\n")
        f.write(report_text)
    print(f"      Saved text report   -> {paths['report_file'].name}")

    # Save classes list
    with open(paths["classes_file"], "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4)
    print(f"      Saved classes list  -> {paths['classes_file'].name}")

    # Generate and save Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title("Confusion Matrix — Announcement Classifier (Held-out Test Set)", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(paths["matrix_file"], dpi=300)
    plt.close()
    print(f"      Saved confusion matrix plot -> {paths['matrix_file'].name}")

    # 6. Save fitted Pipeline with joblib
    print("\n[6/6] Serializing trained model Pipeline...")
    joblib.dump(pipeline, paths["model_file"])
    print(f"      Saved model Pipeline -> {paths['model_file'].name}")

    print("\n" + "=" * 70)
    print("  TRAINING & EVALUATION SUCCESSFULLY COMPLETED!")
    print("=" * 70)

    return metrics_data


if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        print(f"\n[ERROR] Model training failed: {e}", file=sys.stderr)
        sys.exit(1)
