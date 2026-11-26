"""Explore the ticket data with SQL and save charts to docs/img/.

Run from the project root, after ingest.py:  python src/analysis.py
"""

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to image files, no window
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "tickets.db"
IMG_DIR = ROOT / "docs" / "img"

# One colour for every bar: each chart shows a single series.
BAR = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

QUERIES = {
    "by_category": """
        SELECT category, COUNT(*) AS tickets
        FROM tickets
        GROUP BY category
        ORDER BY tickets DESC
    """,
    "top_intents": """
        SELECT intent, category, COUNT(*) AS tickets
        FROM tickets
        GROUP BY intent
        ORDER BY tickets DESC
        LIMIT 15
    """,
    # Every intent has ~1,000 tickets, so a category's size = how many intents it holds
    "intents_per_category": """
        SELECT category, COUNT(DISTINCT intent) AS intents
        FROM tickets
        GROUP BY category
        ORDER BY intents DESC, category
    """,
    # Word count = number of spaces + 1 (good enough for short messages)
    "length_by_category": """
        SELECT category,
               ROUND(AVG(LENGTH(message) - LENGTH(REPLACE(message, ' ', '')) + 1), 1) AS avg_words,
               MIN(LENGTH(message) - LENGTH(REPLACE(message, ' ', '')) + 1) AS min_words,
               MAX(LENGTH(message) - LENGTH(REPLACE(message, ' ', '')) + 1) AS max_words
        FROM tickets
        GROUP BY category
        ORDER BY avg_words DESC
    """,
    "word_counts": """
        SELECT LENGTH(message) - LENGTH(REPLACE(message, ' ', '')) + 1 AS words
        FROM tickets
    """,
}


def style(ax, title, xlabel):
    """Shared look: quiet axes and grid so the bars carry the message."""
    ax.set_title(title, loc="left", fontsize=13, color=TEXT, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def bar_chart(labels, values, title, xlabel, filename):
    """Horizontal bars, largest at the top, each labelled with its value."""
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(labels) + 1.2), facecolor=SURFACE)
    bars = ax.barh(labels[::-1], values[::-1], color=BAR, height=0.7)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values[::-1]], padding=4, color=TEXT, fontsize=9)
    style(ax, title, xlabel)
    ax.set_xlim(0, max(values) * 1.12)  # room for the value labels
    fig.tight_layout()
    fig.savefig(IMG_DIR / filename, dpi=150)
    plt.close(fig)


def histogram(words, filename):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    # Bin edges at 0.5, 1.5, 2.5 ... so each bar is centred on a whole word count
    edges = [n + 0.5 for n in range(words.max() + 1)]
    ax.hist(words, bins=edges, color=BAR, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(1, words.max() + 1))
    style(ax, "How long are customer messages?", "Words per message")
    ax.set_ylabel("Tickets", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axvline(words.median(), color=TEXT, linewidth=1, linestyle="--")
    ax.text(words.median() + 0.4, ax.get_ylim()[1] * 0.92, f"median {words.median():.0f} words", color=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG_DIR / filename, dpi=150)
    plt.close(fig)


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        r = {name: pd.read_sql(sql, conn) for name, sql in QUERIES.items()}

    total = r["by_category"]["tickets"].sum()
    r["by_category"]["share_%"] = (100 * r["by_category"]["tickets"] / total).round(1)

    print(f"Total tickets: {total:,}\n")
    print("Tickets by category:\n", r["by_category"].to_string(index=False), "\n")
    print("Top 15 intents:\n", r["top_intents"].to_string(index=False), "\n")
    print("Intents per category:\n", r["intents_per_category"].to_string(index=False), "\n")
    print("Message length (words) by category:\n", r["length_by_category"].to_string(index=False), "\n")

    bar_chart(r["by_category"]["category"].tolist(), r["by_category"]["tickets"].tolist(),
              "Tickets by category", "Tickets", "tickets_by_category.png")
    bar_chart(r["intents_per_category"]["category"].tolist(), r["intents_per_category"]["intents"].tolist(),
              "Intents per category (each intent has ~1,000 tickets)", "Distinct intents", "intents_per_category.png")
    histogram(r["word_counts"]["words"], "message_length.png")
    print(f"Charts saved to {IMG_DIR}")


if __name__ == "__main__":
    main()
