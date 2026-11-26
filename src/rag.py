"""RAG: find the help articles that match a ticket, then have Claude draft a reply that cites them.

RAG = Retrieval-Augmented Generation:
  1. Retrieval:  turn every article and the ticket into vectors (embeddings); the closest vectors win.
  2. Generation: give Claude the ticket plus only those articles, and ask for a reply based on them.

Run from the project root:  python src/rag.py
"""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classifier  # noqa: E402
import llm  # noqa: E402

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small (~90 MB), free, runs locally

SYSTEM_PROMPT = """You are a customer support agent for an online store. Draft a reply to the customer's ticket.

Rules:
- Use ONLY the facts in the help articles provided. Do not invent policies, prices, times or steps.
- If the articles don't answer the question, say a colleague will follow up, and don't guess.
- Be warm and concise: at most 120 words. Address the customer directly. No subject line.
- Placeholders like {{Order Number}} stand for real values; keep them as they are.
- On the last line, write "Sources:" followed by the exact titles of the articles you used, separated by "; "."""

_embedder = None
_articles = None
_vectors = None


def load_articles():
    """Read each markdown file into a dict: title, category, intents and body text."""
    articles = []
    for path in sorted(KB_DIR.glob("*.md")):
        _, header, body = path.read_text(encoding="utf-8").split("---", 2)
        meta = dict(line.split(": ", 1) for line in header.strip().splitlines())
        articles.append({
            "file": path.name,
            "title": meta["title"],
            "category": meta["category"],
            "intents": [i.strip() for i in meta["intents"].split(",")],
            "body": body.strip(),
        })
    return articles


def _index():
    """Embed all articles once and keep them in memory (22 articles: no database needed)."""
    global _embedder, _articles, _vectors
    if _vectors is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
        _articles = load_articles()
        # Normalised vectors: the dot product of two of them is their cosine similarity (-1 to 1)
        _vectors = _embedder.encode([a["body"] for a in _articles], normalize_embeddings=True)
    return _embedder, _articles, _vectors


def retrieve(text, k=3):
    """The k articles most similar in meaning to the text, best first, each with a similarity score."""
    embedder, articles, vectors = _index()
    query = embedder.encode([text], normalize_embeddings=True)[0]
    scores = vectors @ query
    best = np.argsort(scores)[::-1][:k]
    return [{**articles[i], "score": float(scores[i])} for i in best]


def draft_reply(ticket, k=3, model=llm.OPUS):
    """Retrieve the top articles and ask Claude for a reply grounded in them."""
    found = retrieve(ticket, k)
    articles_block = "\n\n".join(f'<article title="{a["title"]}">\n{a["body"]}\n</article>' for a in found)

    response = llm.client().beta.messages.create(
        model=model,
        max_tokens=2000,
        output_config={"effort": "medium"},
        # If the model declines a request for safety reasons, Anthropic re-runs it on another model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<articles>\n{articles_block}\n</articles>\n\n<ticket>\n{ticket}\n</ticket>"}],
    )
    if response.stop_reason == "refusal":
        return {"reply": None, "sources": [], "retrieved": found, "cost": llm.cost(model, response.usage)}

    reply = next(b.text for b in response.content if b.type == "text").strip()
    # Check the citations: every cited title must be one of the articles we actually gave it
    last_line = reply.splitlines()[-1]
    cited = [t.strip() for t in last_line.removeprefix("Sources:").split(";")] if last_line.startswith("Sources:") else []
    given = {a["title"] for a in found}
    return {
        "reply": reply,
        "sources": cited,
        "sources_valid": bool(cited) and all(c in given for c in cited),
        "retrieved": found,
        "cost": llm.cost(model, response.usage),
    }


def evaluate_retrieval():
    """On every test ticket: is an article for the ticket's intent ranked 1st? In the top 3? (Free: runs locally.)"""
    _, test_df = classifier.split(classifier.load_tickets())
    embedder, articles, vectors = _index()
    queries = embedder.encode(test_df["message"].tolist(), normalize_embeddings=True, batch_size=256)
    ranked = np.argsort(queries @ vectors.T, axis=1)[:, ::-1]

    hit1 = hit3 = 0
    for intent, order in zip(test_df["intent"], ranked):
        top = [intent in articles[i]["intents"] for i in order[:3]]
        hit1 += top[0]
        hit3 += any(top)
    n = len(test_df)
    print(f"Retrieval on {n:,} test tickets: right article 1st {hit1 / n:.1%}, in top 3 {hit3 / n:.1%}")


if __name__ == "__main__":
    evaluate_retrieval()
    examples = [
        "you charged my card twice this month!!! fix it",
        "I moved last week, please make sure my next order goes to the new place",
        "how long do I have to send back a jacket I dont like",
    ]
    total = 0.0
    for ticket in examples:
        result = draft_reply(ticket)
        total += result["cost"]
        print("\n" + "=" * 80)
        print("TICKET:   ", ticket)
        print("RETRIEVED:", ", ".join(f'{a["title"]} ({a["score"]:.2f})' for a in result["retrieved"]))
        print("CITATIONS VALID:", result.get("sources_valid"))
        print("-" * 80)
        print(result["reply"])
    print(f"\nDrafting cost for {len(examples)} replies: ${total:.4f}")
