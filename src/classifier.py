"""Train and evaluate a TF-IDF + logistic regression model that predicts a ticket's category.

Run from the project root, after ingest.py:  python src/classifier.py
Other code uses predict(text) to classify a new ticket with the saved model.
"""

import sqlite3
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "tickets.db"
MODEL_PATH = ROOT / "models" / "category_model.joblib"
IMG_DIR = ROOT / "docs" / "img"

TEST_SIZE = 0.2      # 20% of tickets held back for testing
RANDOM_STATE = 42    # fixed seed: the same split every run (Session 3 reuses it)


def load_tickets():
    """All tickets, with duplicate messages removed so a test ticket never has a twin in training."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT id, message, category, intent FROM tickets", conn)
    key = df["message"].str.lower().str.strip()
    return df[~key.duplicated()].reset_index(drop=True)


def split(df):
    """Stratified split: every category keeps the same share in train and test."""
    return train_test_split(df, test_size=TEST_SIZE, stratify=df["category"], random_state=RANDOM_STATE)


def build_model():
    """Two steps chained together: text -> word-weight numbers -> category."""
    return Pipeline([
        # Turns each message into numbers: one per word or word pair, weighted by how distinctive it is
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        # Learns which words point to which category
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def save_confusion_matrix(y_true, y_pred, labels):
    """Rows = true category, columns = predicted. Shading = share of that row, so small categories show up too."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_share = cm / cm.sum(axis=1, keepdims=True)
    blues = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#2a78d6", "#0d366b"])

    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor="#fcfcfb")
    ax.imshow(row_share, cmap=blues, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if row_share[i, j] > 0.5 else "#0b0b0b")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", color="#52514e")
    ax.set_yticks(range(len(labels)), labels, color="#52514e")
    ax.set_xlabel("Predicted category", color="#52514e")
    ax.set_ylabel("True category", color="#52514e")
    ax.set_title("Confusion matrix: ML model on the test set", loc="left", fontsize=13, pad=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return cm


def train():
    df = load_tickets()
    train_df, test_df = split(df)
    print(f"{len(df):,} unique tickets -> {len(train_df):,} train / {len(test_df):,} test\n")

    model = build_model()
    model.fit(train_df["message"], train_df["category"])
    y_pred = model.predict(test_df["message"])

    print(f"Accuracy: {accuracy_score(test_df['category'], y_pred):.3f}\n")
    print(classification_report(test_df["category"], y_pred, digits=3))

    labels = sorted(df["category"].unique())
    cm = save_confusion_matrix(test_df["category"], y_pred, labels)

    # List the most common mistakes in words, easier to read than the matrix
    mistakes = [(cm[i, j], labels[i], labels[j])
                for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j]]
    print("Most common mistakes (true -> predicted):")
    for n, true, pred in sorted(mistakes, reverse=True)[:8]:
        print(f"  {n:>3}  {true} -> {pred}")
    results = test_df.assign(predicted=y_pred)
    wrong = results[results["category"] != results["predicted"]]
    print("Examples:")
    for _, row in wrong.head(8).iterrows():
        print(f"  [{row['category']} -> {row['predicted']}] {row['message'][:85]}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


_model = None


def predict(text):
    """Classify one ticket. Returns (category, confidence between 0 and 1)."""
    global _model
    if _model is None:
        # joblib can run code hidden in a file, so only load models we trained ourselves (train() above).
        # The file lives in models/, which is gitignored: it is never downloaded or shared.
        _model = joblib.load(MODEL_PATH)  # load once, reuse for later calls
    probabilities = _model.predict_proba([text])[0]
    best = probabilities.argmax()
    return _model.classes_[best], float(probabilities[best])


if __name__ == "__main__":
    train()
