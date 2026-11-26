"""Classify tickets with LLMs (Claude Opus, Claude Haiku, a local model) and compare them with the ML model.

Two test sets:
  clean: 200 tickets from the Session 2 test split (same style as the training data)
  messy: 44 hand-written realistic tickets (typos, slang, two issues in one) in data/messy_tickets.csv

Every prediction is stored in the SQLite table `predictions`. Re-running skips tickets already done,
so we never pay twice for the same answer.

Run from the project root:  python src/llm_classifier.py [methods...]   e.g.  python src/llm_classifier.py ml haiku
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classifier  # noqa: E402
import llm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "docs" / "results.md"
LOCAL_MODEL = "qwen3:8b"
BUDGET_PER_RUN = 2.50  # US dollars: stop calling paid models if one run reaches this

# One line per category, built from the dataset's intents: the same information a human router would get
CATEGORIES = {
    "ACCOUNT": "creating, editing, switching type, deleting an account; passwords; sign-up problems",
    "CANCEL": "fees for cancelling or ending a plan or membership early",
    "CONTACT": "how to reach customer service or a human agent",
    "DELIVERY": "delivery options, methods, countries and how long delivery takes",
    "FEEDBACK": "complaints, praise and reviews",
    "INVOICE": "finding, downloading or correcting invoices and receipts",
    "ORDER": "placing, changing, cancelling or tracking an order",
    "PAYMENT": "payment methods and payment problems (declined, double charges)",
    "REFUND": "return and refund policy, getting a refund, tracking a refund",
    "SHIPPING": "setting up or changing a shipping address",
    "SUBSCRIPTION": "subscribing to or unsubscribing from the newsletter",
}

PROMPT = "Classify the customer support ticket into exactly one category.\n\nCategories:\n" + "\n".join(
    f"- {name}: {desc}" for name, desc in CATEGORIES.items()
) + "\n\nIf a ticket mentions several issues, choose the one the customer most needs solved."

# Structured output: the model must return {"category": <one of the 11>}
SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": list(CATEGORIES)}},
    "required": ["category"],
    "additionalProperties": False,
}


def classify_claude(text, model):
    """Returns (category, cost in $). Category is None if the model declined."""
    response = llm.client().messages.create(
        model=model,
        max_tokens=1000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    spent = llm.cost(model, response.usage)
    if response.stop_reason == "refusal":
        return None, spent
    answer = next(b.text for b in response.content if b.type == "text")
    return json.loads(answer)["category"], spent


def classify_local(text):
    """Same prompt and schema, answered by a model running on this PC through Ollama. Free."""
    import ollama

    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
        format=SCHEMA,
        think=False,
        options={"temperature": 0},
    )
    return json.loads(response.message.content)["category"], 0.0


METHODS = {
    "ml": ("ML: TF-IDF + logistic regression", lambda t: (classifier.predict(t)[0], 0.0)),
    "haiku": ("Claude Haiku 5.5", lambda t: classify_claude(t, llm.HAIKU)),
    "opus": ("Claude Opus 5.5", lambda t: classify_claude(t, llm.OPUS)),
    "local": (f"Local: {LOCAL_MODEL} (Ollama)", classify_local),
}


def test_sets():
    _, test_df = classifier.split(classifier.load_tickets())
    clean, _ = train_test_split(test_df, train_size=200, stratify=test_df["category"], random_state=42)
    messy = pd.read_csv(ROOT / "data" / "messy_tickets.csv")
    return {"clean": clean[["message", "category"]], "messy": messy[["message", "category"]]}


def run(method_keys):
    sets = test_sets()
    with sqlite3.connect(classifier.DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            method TEXT, dataset TEXT, message TEXT, true_category TEXT,
            predicted TEXT, seconds REAL, cost_usd REAL,
            PRIMARY KEY (method, dataset, message))""")
        spent = 0.0
        for key in method_keys:
            label, classify = METHODS[key]
            for dataset, df in sets.items():
                done = {r[0] for r in conn.execute(
                    "SELECT message FROM predictions WHERE method=? AND dataset=?", (label, dataset))}
                todo = df[~df["message"].isin(done)]
                print(f"{label} / {dataset}: {len(done)} done, {len(todo)} to go")
                for message, true_category in zip(todo["message"], todo["category"]):
                    if spent >= BUDGET_PER_RUN:
                        print(f"Budget of ${BUDGET_PER_RUN} reached, stopping.")
                        return
                    start = time.perf_counter()
                    predicted, call_cost = classify(message)
                    seconds = time.perf_counter() - start
                    spent += call_cost
                    conn.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
                                 (label, dataset, message, true_category, predicted, seconds, call_cost))
                    conn.commit()
        print(f"Spent this run: ${spent:.3f}")


def report():
    """Build the comparison table from the stored predictions and save it to docs/results.md."""
    with sqlite3.connect(classifier.DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM predictions", conn)
    df["correct"] = df["predicted"] == df["true_category"]
    rows = []
    for method, g in df.groupby("method", sort=False):
        clean, messy = g[g["dataset"] == "clean"], g[g["dataset"] == "messy"]
        rows.append({
            "Method": method,
            "Accuracy, clean (n=200)": f"{clean['correct'].mean():.1%}" if len(clean) else "-",
            "Accuracy, messy (n=44)": f"{messy['correct'].mean():.1%}" if len(messy) else "-",
            "Cost per 1,000 tickets": f"${g['cost_usd'].mean() * 1000:.2f}",
            "Median seconds per ticket": f"{g['seconds'].median():.3f}",
        })
    headers = list(rows[0])
    table = "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(r.values()) + " |" for r in rows]
    )

    wrong = df[(df["dataset"] == "messy") & ~df["correct"]]
    mistakes = "\n".join(f"- **{r.method}**: \"{r.message}\": expected {r.true_category}, got {r.predicted}"
                         for r in wrong.itertuples())
    RESULTS_PATH.write_text(
        "# ML vs LLM: ticket classification results\n\n"
        "Generated by `src/llm_classifier.py`. Accuracy = share of tickets put in the right category.\n\n"
        f"{table}\n\n## Mistakes on the messy set\n\n{mistakes or 'None.'}\n",
        encoding="utf-8",
    )
    print(table)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    keys = sys.argv[1:] or list(METHODS)
    run(keys)
    report()
