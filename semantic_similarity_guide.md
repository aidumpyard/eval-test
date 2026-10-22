# Semantic Similarity — Developer Guide

## Why this exists

When an AI system produces text output, you need to answer two different questions at different stages of work:

- **During development:** "Did my prompt change make things better or worse?"
- **In production:** "Can I prove what this system produced, and is it degrading over time?"

These are different questions and need different tools. This guide covers both.

---

## Core concept: what is semantic similarity?

Two sentences can mean the same thing without sharing any words:
> "The account was suspended." vs "Access has been revoked."

Semantic similarity measures how close two pieces of text are in *meaning*, not in exact wording.

The challenge for AI outputs is scale: your outputs are often long (hundreds of words), and most similarity tools were designed for short sentences. The tools in this guide are chosen specifically because they work on large text.

---

## The two-question mental model

```
DEVELOPMENT                          PRODUCTION
-----------                          ----------
Question: did I regress?             Question 1: what did the system produce at time T?
Tool: ROUGE-L (dev_check.py)         Tool: SHA-256 hash log (prod_monitor.py)
Speed: milliseconds, no model        Speed: microseconds, no model

                                     Question 2: is the system drifting?
                                     Tool: BERTScore (prod_monitor.py)
                                     Speed: seconds, requires torch
```

Never use the same metric for both stages. Dev tools prioritise speed. Production tools prioritise auditability and precision.

---

## Tool 1: ROUGE-L (development smoke test)

**File:** `dev_check.py`
**Install:** `pip install rouge-score`

### What it measures

ROUGE-L measures the longest sequence of words that appears in both texts, in the same order. It does not require any ML model — it is pure string matching.

Example:
- Reference: *"The customer account was suspended due to missed payments."*
- Candidate: *"Account suspended after missed payments."*
- ROUGE-L F1: ~0.62 — significant overlap, same meaning, acceptable

### When to use it

Run it every time you change a prompt. Takes under a second. Compare against a stored baseline score for each example.

### Verdict logic

| Situation | Verdict |
|---|---|
| Score dropped ≥ 0.15 from baseline | FAIL — stop and investigate |
| Score dropped 0.05–0.15 from baseline | WARN — check a few outputs manually |
| Score stable or improved | PASS |
| No baseline, score < 0.30 | FAIL — output is structurally different |

### Usage

```python
from dev_check import smoke_test, run_suite

# Single check
result = smoke_test(
    reference="The customer account was suspended due to missed payments.",
    candidate="Account suspended after missed payments.",
    baseline=0.60,   # your previously recorded score for this example
)
print(result.verdict)   # PASS / WARN / FAIL
print(result.rouge_l)   # e.g. 0.62
print(result.notes)     # list of observations

# Batch check (recommended — run your whole example suite)
examples = [
    {
        "label": "suspension notice",
        "reference": "The customer account was suspended due to missed payments.",
        "candidate": "Account suspended after missed payments.",
        "baseline": 0.60,
    },
    # add more examples...
]
suite = run_suite(examples, fail_threshold=3)
print(suite["summary"])   # e.g. "PASS — 1/5 examples flagged (threshold: 3)"
```

### What ROUGE-L does NOT catch

- Factual errors that use the same words in a different order
- Hallucinated content added after a correct opening
- Subtle meaning changes ("low risk" vs "high risk")

For those, use BERTScore in production or a human review step.

---

## Tool 2: Hash logging (production — audit trail)

**File:** `prod_monitor.py`
**Install:** none (standard library only)

### What it does

Creates a fingerprint of every inference call. The fingerprint contains:
- A SHA-256 hash of the input (proves exactly what was sent)
- A SHA-256 hash of the output (proves exactly what was received)
- Model name, pinned version, temperature, timestamp

This answers the question: *"At 14:32 on Tuesday, exactly what did the model produce for this input?"*

### Critical rule

**Always pin the model version.** Never use `"latest"` in production. A model update changes outputs silently — the hash log is the only way to detect this after the fact.

### Usage

```python
from prod_monitor import log_inference, verify_inference

# Log every inference call
record = log_inference(
    model="gpt-4o",
    model_version="2024-11-20",   # pinned — never "latest"
    temperature=0.0,
    input_text=prompt,
    output_text=response,
)
# Store record in your DB / logging system / S3

# Later: verify a stored record matches specific texts
result = verify_inference(
    record=record,
    input_text=original_prompt,
    output_text=disputed_output,
)
print(result["both_match"])   # True = confirmed match, False = tampered or different
```

### Why temperature=0 matters

At temperature=0, most models are deterministic — the same input produces the same output every time. This is what makes the hash reproducible. At temperature > 0, outputs vary on each call. If you must use temperature > 0, log the output hash anyway — it still proves what was produced even if you cannot regenerate it identically.

---

## Tool 3: BERTScore (production — drift detection)

**File:** `prod_monitor.py`
**Install:** `pip install bert-score` (downloads ~700 MB torch on first run — one-time cost)

### What it measures

BERTScore matches each word in the candidate output to the most similar word in the reference output, using a language model. It returns Precision, Recall, and F1. You primarily use **F1**.

Unlike ROUGE-L, it understands synonyms and paraphrases. Unlike cosine similarity on a single vector, it works correctly on long text because it operates at the token level across the full document.

### When to use it

Run on your eval set (15–30 representative examples) after every production deploy, or on a daily schedule. You are tracking the *trend* of mean F1 over time, not reacting to individual scores.

**Signal rule:** a downward trend across 3 or more consecutive runs warrants investigation. A single drop does not.

### Usage

```python
from prod_monitor import score_outputs

# Run against your eval set
report = score_outputs(
    references=["Expected output 1", "Expected output 2", "Expected output 3"],
    candidates=["Actual output 1",   "Actual output 2",   "Actual output 3"],
    baseline_mean_f1=0.87,   # your last known-good score; omit on first run
)

print(report.verdict)       # STABLE / WARN / DEGRADED
print(report.mean_f1)       # e.g. 0.85
print(report.delta)         # e.g. -0.02 (dropped from baseline)
print(report.notes)         # human-readable observations
print(report.per_pair_f1)   # individual scores — inspect the lowest ones
```

### Interpreting scores

BERTScore F1 is not a percentage. Use it relatively, not absolutely.
- Track the trend, not the number
- A drop of 0.03+ from your baseline is meaningful
- A drop of 0.01–0.02 is noise unless it persists

---

## How the two files work together

```
You change a prompt
        │
        ▼
dev_check.py — ROUGE-L smoke test
  FAIL/WARN → fix before deploying
  PASS ──────────────────────────────────────────────────────────────────┐
                                                                         │
                                                                         ▼
                                                               Deploy to production
                                                                         │
                                                         ┌───────────────┴──────────────────┐
                                                         ▼                                  ▼
                                              prod_monitor.log_inference()      prod_monitor.score_outputs()
                                              (every call, real-time)           (scheduled, on eval set)
                                              Stores hash + metadata            Tracks F1 trend
                                                         │                                  │
                                                         ▼                                  ▼
                                              Incident: verify_inference()       Alert: DEGRADED verdict
                                              "Did the system produce this?"     "Quality is dropping"
```

---

## Getting started: step-by-step

**Step 1 — Install dependencies**
```bash
pip install rouge-score          # dev_check.py
pip install bert-score           # prod_monitor.py (large download)
```

**Step 2 — Build your example set**
Collect 15–20 real input/output pairs from your system that represent good outputs. These become your smoke-test suite and your BERTScore eval set.

**Step 3 — Record baselines**
Run `smoke_test` on each example and record the ROUGE-L score. Run `score_outputs` and record the mean F1. These are your baselines.

**Step 4 — Wire into your workflow**
- Call `run_suite` in your development workflow before merging any prompt change
- Call `log_inference` in production for every model call
- Run `score_outputs` on a schedule (daily or per deploy) against your eval set

**Step 5 — Calibrate thresholds**
The default thresholds (ROUGE-L drop of 0.05 for WARN, BERTScore drop of 0.03 for DEGRADED) are starting points. After a few weeks of runs, adjust them to match the natural variance you observe in your specific outputs.

---

## Limitations to communicate to stakeholders

- **ROUGE-L** is a fast proxy, not a quality measurement. It catches structural drift, not factual errors.
- **BERTScore** catches semantic drift but cannot tell you *why* quality dropped — it tells you to investigate, not what to fix.
- **Neither tool** replaces human review for high-stakes outputs. They reduce the volume of outputs a human needs to review by surfacing the right ones.
- Scores across different tasks are not comparable. A ROUGE-L of 0.60 on summarisation tasks and 0.60 on Q&A tasks mean different things. Always compare within the same task type.
