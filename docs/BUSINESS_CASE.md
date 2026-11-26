# Business Case: AI-Assisted Support Ticket Triage

*Status: first draft (Session 1). The impact estimate is completed in Session 4 with measured results.*

## The problem

Every incoming support ticket has to be **read, categorised and routed** to the right team before anyone can help
the customer. In many support teams this triage is done by hand, and it causes three problems:

1. **It's slow.** An agent spends time reading and routing each ticket before any real work starts, and during
   busy periods tickets wait in a shared queue.
2. **It's inconsistent.** Different agents label the same kind of ticket differently. Misrouted tickets bounce
   between teams, which adds delay and frustrates the customer.
3. **It doesn't scale.** More tickets means more triage hours. That time is spent on sorting, not solving.

On top of routing, agents often write replies from scratch, even though most questions (refunds, invoices,
delivery times) are answered by existing help articles.

## Who is affected

| Who | How |
|---|---|
| **Customers** | Wait longer for a first useful answer; get passed between teams |
| **Support agents** | Spend part of every shift on repetitive sorting and retyping standard answers |
| **Team leads** | Can't easily see what customers are contacting them about, or plan staffing around it |
| **The business** | Pays for triage hours instead of problem-solving, and risks lower customer satisfaction |

## Proposed solution

An assistant that, for each new ticket:

1. **Classifies** it into a category (e.g. REFUND, INVOICE, DELIVERY).
2. **Routes** it to the team that owns that category.
3. **Drafts a reply** grounded in the relevant help articles and says which article it used, so the agent
   reviews and edits instead of writing from scratch.

A human stays in the loop: the agent checks the category and the draft before anything goes to the customer.

## What we will measure

| Question | Metric | When |
|---|---|---|
| How often is the category right? | Accuracy, plus precision, recall and F1 **per category** | Session 2 (ML), Session 3 (LLM) |
| Which categories get confused? | Confusion matrix | Session 2 |
| Do we need an LLM, or does a simple model do the job? | ML vs cloud LLM vs local LLM on accuracy, cost and speed | Session 3 |
| What does it cost to run? | Cost per 1,000 tickets | Session 3 |
| How fast is it? | Seconds per ticket | Session 3 |
| What is it worth? | Estimated agent hours saved per week (formula below) | Session 4 |

**Per-category metrics matter here** because the categories are imbalanced: ACCOUNT is 22% of tickets, CANCEL only
3.5%. A model could ignore small categories completely and still score high overall accuracy.

## Impact estimate (method; numbers come in Session 4)

```
hours saved per week = tickets per week
                     × triage minutes saved per ticket
                     × share of tickets routed correctly
                     ÷ 60
```

Every input will be stated as an explicit assumption, and the "share routed correctly" will come from our measured
test results, not a guess. Misrouted tickets still need manual handling, so they count as zero savings.

## What the data can and can't tell us

We use the public **Bitext customer support dataset**: 26,872 synthetic customer messages labelled with
11 categories and 27 intents. No real customer data is used.

**Findings from the Session 1 analysis** (charts in [`img/`](img/)):

- **Category sizes are uneven, but by design.** Every intent has ~1,000 examples; ACCOUNT is the largest category
  only because it groups 6 intents, while CANCEL and SUBSCRIPTION have 1 each.
- **Messages are short and clean:** median 9 words, maximum 16, each about a single topic.

**Limitation:** real tickets are longer, messier, often cover several issues and have uneven volumes. Results on
this dataset are an **optimistic upper bound**. Before a real rollout, the first step would be to test on a sample
of the team's own historical tickets and interview agents about where triage actually costs them time.
