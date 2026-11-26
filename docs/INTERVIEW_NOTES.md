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
