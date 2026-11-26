# Interview Notes

Running log of decisions, trade-offs, results and problems. Updated at the end of every session.

---

## Session 1: Setup, data and analysis

### What was built
- `src/ingest.py`: downloads the Bitext dataset from Hugging Face and loads it into a SQLite `tickets` table.
  Safe to re-run (rebuilds the table).
- `src/analysis.py`: SQL queries for volume by category and intent, intents per category, and message length.
  Saves 3 charts to `docs/img/`.
- `docs/BUSINESS_CASE.md`: first draft covering the problem, who it affects, and what we measure.

### Decisions and trade-offs

**SQLite instead of PostgreSQL**
- SQLite is a library inside the program; the whole database is one file. No server, port, users or passwords.
  It's built into Python's standard library.
- PostgreSQL is client–server: a separate service that handles many simultaneous users and writers.
- Chose SQLite because the project has one app and ~27k rows, and anyone cloning the repo can run it with zero
  setup. **In production with many concurrent writers I'd move to PostgreSQL;** the SQL is nearly identical.

**Python 3.14 only**
- Removed an older 3.10 install so there's one Python on the machine. The project runs in its own virtual
  environment (`.venv`), so its packages don't clash with anything else.

**Secrets kept out of the repo from day one**
- `.env` holds API keys and is gitignored; `.env.example` shows which settings exist with empty values.
- Tested that git really ignores `.env` before writing any code that uses keys.

**Requirements grow with the project**
- `requirements.txt` only lists what's used so far (pandas, datasets, matplotlib), so every dependency has a reason.

**Three approaches to compare later (ML, cloud LLM, local LLM)**
- *ML model:* TF-IDF + logistic regression trained on our labelled tickets. Tiny, trains in seconds on a CPU,
  near-zero cost, but only knows word patterns it has seen.
- *Cloud LLM:* a large general model called over an API. Not trained by us; given an instruction. Best language
  understanding, but costs per call and ticket text leaves the company.
- *Local LLM:* an open-weight model (~7–8B parameters) run on our own GPU through Ollama. Free and private, usually
  less accurate. Key talking point: **what accuracy do you give up to keep data in-house?**

### Results

| Fact | Value |
|---|---|
| Tickets | 26,872 |
| Categories / intents | 11 / 27 |
| Largest category | ACCOUNT, 5,986 (22.3%) |
| Smallest categories | SUBSCRIPTION 999 (3.7%), CANCEL 950 (3.5%) |
| Tickets per intent | ~1,000 each, almost exactly |
| Message length | median 9 words, range 1–16 |

### Insights worth saying out loud
- **The dataset is synthetic, and the analysis shows it.** Every intent has ~1,000 examples; real ticket volumes are
  never that even. Category imbalance comes only from how many intents each category groups.
- **Results will be an optimistic upper bound.** Messages are short, clean and single-topic; real tickets aren't.
- **Imbalance changes how we evaluate.** A model ignoring CANCEL entirely could still score ~96% accuracy, so
  Session 2 reports metrics per category.

### Problems hit and how they were fixed
- **An uninformative chart.** The first draft charted the top 15 intents; all 15 bars were ~1,000. Replaced it with
  "intents per category", which explains the imbalance. Lesson: look at the output, don't just generate it.
- **Misleading histogram bins.** Bars sat between tick marks, so "9 words" looked like 9–10. Shifted bin edges by
  0.5 so each bar is centred on its word count.
- **`python` opened the Microsoft Store.** Windows ships a `python.exe` shortcut that came first on PATH. The project
  always calls `.venv\Scripts\python.exe` directly, so it isn't affected.
- **Inline Python broke in PowerShell.** Windows PowerShell strips quotes when passing code to programs; moved the
  check into a small script file instead.

### If I did it again
- Talk to the support team first to find their real bottleneck before choosing what to build.

---

## Session 2: Machine learning baseline

### What was built
- `src/classifier.py`: removes duplicate messages, splits 80/20, trains TF-IDF + logistic regression, prints
  per-category metrics and the most common mistakes, saves a confusion matrix and the model. `predict(text)` returns
  a category and a confidence.
- `tests/test_classifier.py`: 4 tests. Three use a tiny made-up dataset, so they run on a fresh clone with no
  database; one checks the saved model and is skipped if it hasn't been trained.

### How the model works (plain version)
- **TF-IDF** turns each message into numbers: one per word and word pair ("money back"), weighted up when the word is
  distinctive (refund) and down when it's everywhere (please, the).
- **Logistic regression** learns a weight per word per category from the training tickets. A new ticket's score for
  each category is the sum of its words' weights, turned into probabilities. The highest wins; that probability is
  the **confidence**.
- Both are chained in one scikit-learn `Pipeline`, so the same text processing is always applied in training and
  prediction.

### Decisions and trade-offs

**Removed duplicates before splitting (data leakage)**
- 2,501 tickets (9%) were near-duplicates; some sentences appeared 10 times. Without removing them, a test ticket
  could have an identical twin in training, so the test would partly measure memory, not learning.
- Safe to remove: no message appeared with two different categories.

**Stratified split with a fixed seed**
- *Stratified:* each category keeps the same share in train and test, so small categories like CANCEL are tested
  properly.
- *Fixed seed (42):* the same split every run, so results are reproducible and Session 3 can test the LLM on exactly
  the same unseen tickets.

**Simple model on purpose**
- A baseline should be cheap and explainable. If it's good enough, the expensive option needs to justify itself.

### Results (test set: 4,875 unseen tickets)

| Metric | Value |
|---|---|
| Accuracy | **99.9%** (7 mistakes) |
| Lowest per-category F1 | 0.997 |
| Training + evaluation time | ~5 seconds on a CPU |
| Model file size | a few MB |

All 7 mistakes involve typos or glued words: "theaddress", "paytment", "adress", "newslettee", "order370795561790".
The model only knows exact words it has seen, so a misspelt key word becomes invisible to it.

### The important caveat: messy tickets
Six realistic tickets written by hand (anecdotal, not a benchmark):

| Ticket | Predicted | Confidence | Should be |
|---|---|---|---|
| "you charged my card twice this month!!! fix it" | ACCOUNT | 31% | PAYMENT |
| "the app keeps logging me out every 5 min" | CANCEL | 41% | ACCOUNT |
| "do u ship to bulgaria" | ACCOUNT | 19% | DELIVERY/SHIPPING |
| "parcel came... box crushed, kettle broken" | DELIVERY | 28% | arguable (REFUND?) |
| "ordered 3 weeks ago... if not by Friday I want my money back" | REFUND | 78% | arguable (two issues) |

**Takeaways to say in an interview:**
- 99.9% says the **dataset is easy** (short, clean, template-like), not that the problem is solved.
- On wording it hasn't seen, the model guesses, **but its confidence drops** (19–41% vs ~99% normally).
  That's useful: **auto-route only above a confidence threshold, send the rest to a human.**
- This is exactly where an LLM might earn its cost: understanding meaning instead of matching words. Session 3
  measures that.

### Problems hit and how they were fixed (Session 2)
- **Leakage from duplicates:** found by checking before training, not after.
- **Security warning on loading the model file:** `joblib` files can run code when loaded. We only load the model we
  trained ourselves, which is never downloaded or shared (gitignored). Documented in the code.

---

## Session 3: Generative AI (RAG and the ML vs LLM comparison)

### What was built
- `data/kb/`: 22 synthetic help articles for one fictional store, covering all 27 intents. One consistent set of
  policies (30-day returns, 5-10 day bank refunds...) so articles never contradict each other.
- `src/llm.py`: one shared Claude client (key loaded from `.env`), model names, and cost per call from token counts.
- `src/rag.py`: embeds the articles, finds the top 3 for a ticket, and has Claude Opus draft a reply that cites them.
- `data/messy_tickets.csv`: 44 hand-written realistic tickets with labels (typos, slang, two issues in one).
- `src/llm_classifier.py`: classifies 200 clean test tickets + the 44 messy ones with four methods, stores every
  prediction in SQLite, and writes the comparison to `docs/results.md`.
- `tests/test_rag.py`: 4 more tests (8 total), none of which call the paid API.

### How RAG works (plain version)
1. **Embeddings:** a small local model (`all-MiniLM-L6-v2`, free, ~90 MB) turns any text into a list of 384 numbers
   that captures its *meaning*. Texts about the same thing get similar numbers even with no words in common
   ("charged twice" ≈ "duplicate payment").
2. **Retrieval:** embed all 22 articles once; for a ticket, embed it and take the 3 closest articles
   (cosine similarity). With 22 articles a plain numpy array is enough; no vector database needed.
3. **Generation:** send Claude the ticket plus only those 3 articles, with rules: use only these facts, don't
   guess, say a colleague will follow up if the answer isn't there, and list the titles used on a "Sources:" line.
4. **Check:** code verifies every cited title is one of the articles actually provided.

Why RAG instead of just asking the LLM? Without the articles it would invent plausible but wrong policies.
RAG keeps answers **grounded** in the company's own documents, **traceable** (sources), and **easy to update**
(edit an article, no retraining).

### Results

**Retrieval (all 4,875 test tickets, free):** the right article is ranked **1st for 70.2%** of tickets and is in
the **top 3 for 91.9%**. That's why we pass 3 articles, not 1: in one example the top hit was wrong but Claude
picked the right two from the three.

**Reply drafts:** 3 of 3 stuck to the articles and cited them correctly. Cost about $0.012 per reply with Opus.

**Classification: ML vs three LLMs**

| Method | Clean (n=200) | Messy (n=44) | Cost per 1,000 tickets | Median seconds per ticket |
|---|---|---|---|---|
| ML: TF-IDF + logistic regression | 99.5% | 56.8% | $0.00 | 0.001 |
| Claude Haiku 5.5 | 99.0% | 93.2% | $0.07 | 1.03 |
| Local: Qwen3 8B (Ollama, RTX 3070) | 94.0% | 86.4% | $0.00 | 0.18 |
| Claude Opus 5.5 | 99.0% | 97.7% | $2.92 | 1.23 |

Total API spend for Session 3: about **$0.77** of the $5 budget.

### What the numbers say (the interview story)
- **On clean, template-like tickets, the free ML model is as good as any LLM.** Paying for an LLM there buys nothing.
- **On realistic messy tickets, the ML model collapses (57%) and every LLM holds up.** The ML model matches words;
  LLMs understand meaning ("charged twice" is a payment problem even without the word "payment").
- **Haiku gives most of Opus's accuracy at about 1/40 of the price.** Opus's extra 4.5 points on the messy set is
  2 tickets out of 44, too few to call it a real difference.
- **The local model is free, private and the fastest LLM here** (no network round trip), but less accurate.
- **A sensible design:** ML first (free, instant); if its confidence is low, send the ticket to Haiku. Most clean
  tickets never cost anything, and the hard ones get an LLM. (Built in Session 4.)

### Caveats I'd mention myself
- **The messy set is small (44) and I wrote and labelled it with AI help.** One ticket = 2.3 points, so small gaps
  are noise. A real evaluation would use a few hundred real tickets labelled by the support team.
- **Some "mistakes" are label quirks, not errors.** In this dataset "delivery *address*" tickets are labelled
  SHIPPING and "shipping *options*" are labelled DELIVERY, the opposite of what the words suggest. The ML model
  learned the quirk from 19k examples; the LLMs only see the category names and descriptions, which explains most
  of the local model's clean-set misses. Opus's two clean "misses" ("when will my package arrive?" -> ORDER) are
  arguably right. **Lesson: clear category definitions matter as much as the model.** Next step would be sharper
  descriptions in the prompt.
- **Tickets with two issues have no single right answer** (e.g. "late, and if not by Friday I want a refund").
  In a real system, ask the support team how they want those routed.

### Decisions and trade-offs
- **Structured outputs:** the API is told the answer must be JSON with a category from a fixed list of 11. No
  parsing of free text, no made-up categories.
- **Low "effort" for classification, medium for replies:** classification is easy, so less reasoning = cheaper and
  faster; replies need more care.
- **Every prediction is saved, and re-runs skip finished tickets:** never pay twice; the comparison can be rebuilt
  at any time. A per-run budget guard ($2.50) stops paid calls if something loops.
- **Refusal fallback only for reply drafts:** if Opus declines a draft, Anthropic retries on another model. Not
  used for the comparison, because a fallback would mix two models into one row of results.
- **Same prompt and schema for every LLM**, so the comparison is fair.
- **Local model: Qwen3 8B**, chosen because it fits fully in the 3070's 8 GB of GPU memory. Stored on G:.

### Problems hit and how they were fixed (Session 3)
- **Writing 22 files in one shell command failed** on a quoting error; nothing had been written, so each was
  written separately and then checked automatically (all 27 intents covered, all cross-references valid).
- **Ollama's background server didn't start** after install; started it directly with the model folder on G:.
- **The local model's 94% looked bad** until the mistakes showed it was the label quirk above. Lesson: read the
  errors before drawing conclusions from a number.
