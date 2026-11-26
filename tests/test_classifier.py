"""Tests for src/classifier.py. Run from the project root:  python -m pytest"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import classifier  # noqa: E402


def tiny_tickets():
    """A small made-up dataset, so tests run without the real database."""
    rows = []
    for i in range(10):
        rows.append({"message": f"I want a refund for order {i}", "category": "REFUND"})
        rows.append({"message": f"where is my package {i} delivery", "category": "DELIVERY"})
        rows.append({"message": f"please send me invoice number {i}", "category": "INVOICE"})
    return pd.DataFrame(rows)


def test_model_learns_simple_patterns():
    df = tiny_tickets()
    model = classifier.build_model()
    model.fit(df["message"], df["category"])
    assert model.predict(["can I get a refund please"])[0] == "REFUND"
    assert model.predict(["my delivery has not arrived, where is the package"])[0] == "DELIVERY"
    assert model.predict(["I need a copy of my invoice"])[0] == "INVOICE"


def test_split_keeps_categories_balanced_and_separate():
    df = tiny_tickets()
    train_df, test_df = classifier.split(df)
    # No ticket ends up in both sets
    assert set(train_df.index).isdisjoint(test_df.index)
    assert len(train_df) + len(test_df) == len(df)
    # Every category appears in the test set (stratified)
    assert set(test_df["category"]) == set(df["category"])


def test_split_is_repeatable():
    df = tiny_tickets()
    first, _ = classifier.split(df)
    second, _ = classifier.split(df)
    assert list(first.index) == list(second.index)


@pytest.mark.skipif(not classifier.MODEL_PATH.exists(), reason="run src/classifier.py first to train the model")
def test_predict_with_saved_model():
    category, confidence = classifier.predict("I would like to get my money back for this order")
    assert category == "REFUND"
    assert 0 <= confidence <= 1
