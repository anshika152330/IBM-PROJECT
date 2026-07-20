"""
Training script for the Fake News Detection System using Linear Regression.

Loads Fake.csv and True.csv, preprocesses text, trains a TF-IDF + Linear
Regression pipeline, evaluates performance, saves artifacts, and generates
visualizations for the Streamlit dashboard.
"""

import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocessing import combine_title_text, preprocess_text

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

FAKE_CSV = DATASET_DIR / "Fake.csv"
TRUE_CSV = DATASET_DIR / "True.csv"
MODEL_PATH = MODELS_DIR / "linear_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
THRESHOLD = 0.5


def load_and_merge_data() -> pd.DataFrame:
    """Load Fake.csv and True.csv, assign labels, and merge into one dataframe."""
    if not FAKE_CSV.exists() or not TRUE_CSV.exists():
        raise FileNotFoundError(
            "Dataset files not found. Place Fake.csv and True.csv in the dataset/ folder."
        )

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    fake_df["label"] = 0  # Fake news
    true_df["label"] = 1  # Real news

    merged_df = pd.concat([fake_df, true_df], ignore_index=True)
    return merged_df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data cleaning steps: null removal, deduplication, and text preprocessing."""
    df = df.copy()

    # Combine title and text for richer features.
    df["combined_text"] = df.apply(
        lambda row: combine_title_text(row.get("title", ""), row.get("text", "")),
        axis=1,
    )

    # Remove null values.
    df = df.dropna(subset=["combined_text", "label"])
    df["combined_text"] = df["combined_text"].astype(str)

    # Remove empty records.
    df = df[df["combined_text"].str.strip().astype(bool)]

    # Remove duplicate records based on combined text.
    df = df.drop_duplicates(subset=["combined_text"], keep="first")

    # Preprocess text.
    df["clean_text"] = df["combined_text"].apply(preprocess_text)
    df = df[df["clean_text"].str.strip().astype(bool)]

    return df.reset_index(drop=True)


def regression_to_binary(predictions: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Convert continuous regression outputs to binary class labels."""
    return (predictions >= threshold).astype(int)


def clip_probability(predictions: np.ndarray) -> np.ndarray:
    """Clip regression outputs to a valid probability range [0, 1]."""
    return np.clip(predictions, 0.0, 1.0)


def evaluate_model(y_true: np.ndarray, y_pred_continuous: np.ndarray) -> dict:
    """Calculate classification and regression metrics."""
    y_pred_binary = regression_to_binary(y_pred_continuous)
    y_prob = clip_probability(y_pred_continuous)

    mse = mean_squared_error(y_true, y_pred_continuous)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred_binary)),
        "precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred_binary, zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred_continuous)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2_score": float(r2_score(y_true, y_pred_continuous)),
        "threshold": THRESHOLD,
        "classification_report": classification_report(
            y_true,
            y_pred_binary,
            target_names=["Fake News", "Real News"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred_binary).tolist(),
        "train_size": None,
        "test_size": None,
        "total_samples": None,
        "fake_count": None,
        "real_count": None,
    }

    return metrics, y_pred_binary, y_prob


def save_pickle(obj, path: Path) -> None:
    """Persist an object using pickle."""
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def plot_label_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Create a bar chart of label distribution."""
    counts = df["label"].value_counts().sort_index()
    labels = ["Fake News (0)", "Real News (1)"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#e74c3c", "#2ecc71"])
    plt.title("Label Distribution in Dataset", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Articles")
    plt.xlabel("Class Label")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height):,}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_metrics(metrics: dict, output_path: Path) -> None:
    """Create a bar chart comparing key classification metrics."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    ]

    plt.figure(figsize=(8, 5))
    colors = ["#3498db", "#9b59b6", "#e67e22", "#1abc9c"]
    bars = plt.bar(metric_names, metric_values, color=colors)
    plt.ylim(0, 1.05)
    plt.title("Model Performance Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_prediction_distribution(y_prob: np.ndarray, output_path: Path) -> None:
    """Create a histogram of predicted probabilities."""
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
    plt.axvline(THRESHOLD, color="#e74c3c", linestyle="--", linewidth=2, label=f"Threshold ({THRESHOLD})")
    plt.title("Prediction Probability Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Probability (Real News)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_heatmap(cm: np.ndarray, output_path: Path) -> None:
    """Create a confusion matrix heatmap."""
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake News", "Real News"],
        yticklabels=["Fake News", "Real News"],
    )
    plt.title("Confusion Matrix Heatmap", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model: LinearRegression, vectorizer: TfidfVectorizer, output_path: Path, top_n: int = 20) -> None:
    """Visualize top TF-IDF features by absolute coefficient magnitude."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_.flatten()
    top_indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]

    top_features = feature_names[top_indices]
    top_values = coefficients[top_indices]
    colors = ["#2ecc71" if value >= 0 else "#e74c3c" for value in top_values]

    plt.figure(figsize=(10, 7))
    plt.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
    plt.title(f"Top {top_n} Feature Importance (Linear Regression Coefficients)", fontsize=14, fontweight="bold")
    plt.xlabel("Coefficient Value")
    plt.ylabel("TF-IDF Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_visualizations(
    df: pd.DataFrame,
    metrics: dict,
    y_prob: np.ndarray,
    model: LinearRegression,
    vectorizer: TfidfVectorizer,
) -> None:
    """Generate and save all required visualizations."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    plot_label_distribution(df, ASSETS_DIR / "label_distribution.png")
    plot_accuracy_metrics(metrics, ASSETS_DIR / "accuracy_graph.png")
    plot_prediction_distribution(y_prob, ASSETS_DIR / "prediction_distribution.png")
    plot_confusion_matrix_heatmap(np.array(metrics["confusion_matrix"]), ASSETS_DIR / "confusion_matrix.png")
    plot_feature_importance(model, vectorizer, ASSETS_DIR / "feature_importance.png")


def train() -> None:
    """Run the complete training pipeline."""
    print("=" * 60)
    print("Fake News Detector - Linear Regression Training")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Loading and merging datasets...")
    df = load_and_merge_data()
    print(f"      Loaded {len(df):,} records.")

    print("\n[2/6] Preprocessing data...")
    df = clean_dataframe(df)
    print(f"      Clean dataset size: {len(df):,} records.")

    print("\n[3/6] TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words="english",
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].values

    print("\n[4/6] Train-test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n[5/6] Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_continuous = model.predict(X_test)
    metrics, _, y_prob = evaluate_model(y_test, y_pred_continuous)

    metrics["train_size"] = int(X_train.shape[0])
    metrics["test_size"] = int(X_test.shape[0])
    metrics["total_samples"] = int(len(df))
    metrics["fake_count"] = int((df["label"] == 0).sum())
    metrics["real_count"] = int((df["label"] == 1).sum())

    print("\n[6/6] Saving model, metrics, and visualizations...")
    save_pickle(model, MODEL_PATH)
    save_pickle(vectorizer, VECTORIZER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    generate_visualizations(df, metrics, y_prob, model, vectorizer)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print(f"MAE       : {metrics['mae']:.4f}")
    print(f"MSE       : {metrics['mse']:.4f}")
    print(f"RMSE      : {metrics['rmse']:.4f}")
    print(f"R² Score  : {metrics['r2_score']:.4f}")
    print("\nClassification Report:\n")
    print(metrics["classification_report"])
    print(f"\nModel saved to       : {MODEL_PATH}")
    print(f"Vectorizer saved to  : {VECTORIZER_PATH}")
    print(f"Metrics saved to     : {METRICS_PATH}")
    print(f"Visualizations saved : {ASSETS_DIR}")


if __name__ == "__main__":
    try:
        train()
    except Exception as error:
        print(f"\nTraining failed: {error}")
        raise
