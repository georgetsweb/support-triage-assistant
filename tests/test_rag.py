"""Tests for src/rag.py and src/llm_classifier.py. None of them call the paid API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import llm_classifier  # noqa: E402
import rag  # noqa: E402


def test_articles_load_with_metadata():
    articles = rag.load_articles()
    assert len(articles) >= 20
    for a in articles:
        assert a["title"] and a["body"]
        assert a["category"] in llm_classifier.CATEGORIES
        assert a["intents"]


def test_retrieval_finds_the_obvious_article():
    titles = [a["title"] for a in rag.retrieve("How do I reset my password? I forgot it.", k=3)]
    assert "Resetting your password" in titles


def test_retrieval_scores_are_sorted():
    scores = [a["score"] for a in rag.retrieve("where is my refund", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_llm_schema_only_allows_known_categories():
    assert llm_classifier.SCHEMA["properties"]["category"]["enum"] == list(llm_classifier.CATEGORIES)
    for name in llm_classifier.CATEGORIES:
        assert name in llm_classifier.PROMPT
