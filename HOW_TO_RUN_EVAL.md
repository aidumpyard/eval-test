# How to Run the JSON Output Eval

Score the JSON your LLM tool produces against a reference ("golden") JSON,
field by field. Nothing here calls an LLM or the network — your tool produces
the output files separately, and these scripts just **read those files off
disk and score them**. Runs are deterministic and replayable.

Files involved:
- [`eval_json.py`](eval_json.py) — the runner you call (API + command line).
- [`dev_check.py`](dev_check.py) — ROUGE-L engine (reused).
- [`prod_monitor.py`](prod_monitor.py) — BERTScore engine (reused, optional).

---

## 1. Install

```bash
pip install rouge-score        # required (ROUGE-L + exact match)
pip install bert-score         # optional — only if you use the bertscore metric
                               # (pulls torch ~700 MB on first install)
```

`exact` and `rouge` need no model and run in milliseconds. `bertscore` loads a
local embedding model onto your machine — still no API call, no data leaves
your computer.

---

## 2. Prepare your files

You need **one reference file** and **one or more output files**, all with the
same schema and a shared id field. A top-level JSON array, or `.jsonl`
(one object per line), both work.

`golden.json` (expected answers):
```json
[
  {
    "id": "case-001",
    "summary": "The customer account was suspended due to missed payments over three months.",
    "risk_assessment": {"decision": "suspend", "level": "high"},
    "risks": ["late payment", "credit exposure", "operational"]
  }
]
```

`run_v1.json` (your tool's output — same `id`s):
```json
[
  {
    "id": "case-001",
    "summary": "Account suspended after three months of missed payments.",
    "risk_assessment": {"decision": "suspend", "level": "high"},
    "risks": ["late payment", "credit exposure"]
  }
]
```

Records are paired by `id`, so order and length may differ between files.

---

## 3. Choose a metric per field

Different field types need different tests. Do **not** use one metric for
everything.

| Field type | Example | Metric | Why |
|---|---|---|---|
| Free text | `summary` | `bertscore` | Rewording is fine; meaning must match. ROUGE-L misses meaning flips. |
| Category / enum | `decision`, `level` | `exact` | Right or wrong. No model needed. |
| Tag list | `risks` | `rouge` or `exact` | Word/overlap proxy, or strict equality. |

Score ranges: `exact` → 1.0 (match) or 0.0 (miss); `rouge` → ROUGE-L F1 in
[0,1]; `bertscore` → F1, read as a **trend vs a baseline**, not an absolute %.
A field's mean score is the average across all records. **Scores from
different fields are not comparable** — never pool them.

---

## 4. Run it — command line

Score several output files against one reference and get a comparison table.
Field syntax is `path` or `path:metric` (dotted paths for nesting):

```bash
python3 eval_json.py \
  --ref   golden.json \
  --out   run_v1.json run_v2.json \
  --fields summary:bertscore risk_assessment.decision:exact risks:rouge \
  --key   id
```

Output:
```
=== Comparison (mean score) ===
  field [metric]                     run_v1.json   run_v2.json
  ------------------------------------------------------------
  summary [bertscore]                     0.9120        0.7840
  risk_assessment.decision [exact]        1.0000        0.5000
  risks [rouge]                           0.9445        0.5000
```

Read it column by column — higher is closer to golden. Here `run_v1` wins:
perfect decisions, and a higher-meaning summary.

Flags:
- `--ref` — reference file (one).
- `--out` — one or more output files (space-separated).
- `--fields` — `path` or `path:metric`. Metrics: `rouge`, `bertscore`, `exact`, `both`.
- `--key id` — pair records by this field. Use `--key ''` to pair by position.
- `--metric rouge` — default metric for any field given as bare `path`.

---

## 5. Run it — Python (per-record detail)

Use the API when you want the per-record breakdown (which `id` failed and why),
not just the means.

```python
from eval_json import evaluate, load_records

report = evaluate(
    candidates=load_records("run_v1.json"),
    references=load_records("golden.json"),
    fields=[
        {"path": "summary",                  "metric": "bertscore"},
        {"path": "risk_assessment.decision", "metric": "exact"},
        {"path": "risk_assessment.level",    "metric": "exact"},
        {"path": "risks", "list_sep": "; ",  "metric": "rouge"},
    ],
    key="id",
)

report.print_summary()                       # verdicts + flagged records
report.mean_for("risk_assessment.decision", "exact")   # -> 1.0

# Inspect individual records:
for fe in report.field_evals:
    for r in fe.per_record:
        print(fe.field_path, r.label, r.score, r.verdict, r.note)
```

`print_summary()` shows, per field: mean score, PASS/WARN/FAIL counts, up to
five flagged records with the reason, and any records skipped because the
field was missing.

Field spec options: `path` (required), `metric` (defaults to the call's
`metric`), `list_sep` (how list values are joined, default newline),
`baseline` (a known-good score for `rouge`/`bertscore` so the verdict is
relative — see below).

---

## 6. Baselines (optional but recommended)

By default `rouge`/`bertscore` verdicts use fixed built-in thresholds that are
**unvalidated starting guesses** (see
[`production_llm_controls.md`](production_llm_controls.md) §8). For real
gating, record a per-field mean from a run you trust and pass it as `baseline`;
the verdict then reports regression *relative to that baseline* instead:

```python
{"path": "summary", "metric": "bertscore", "baseline": 0.91}
```

`exact` needs no baseline — it is right/wrong by definition.

---

## 7. Common issues

| Symptom | Cause / fix |
|---|---|
| `(!) N candidate(s) had no matching reference by 'id'` | Those records have an `id` not in the reference file. Check the id values match. |
| `skipped (field missing): [...]` | That field path was absent or empty on those records. Check the path spelling / nesting. |
| `Positional pairing needs equal lengths` | You used `--key ''` but the files differ in length. Add a shared id and pair with `--key id`. |
| `A field requested BERTScore but ... unavailable` | `pip install bert-score`. |
| `exact` always 0.0 on a field | Values differ by case/whitespace only? `exact` already ignores those. Otherwise the values genuinely differ — inspect the `note`. |

---

## 8. What this does NOT do

- It does not judge correctness beyond matching a reference. A confidently
  wrong output that happens to match golden wording still scores high on
  `rouge`. Use `bertscore` for meaning and `exact` for categories.
- It does not call an LLM, so it cannot explain *why* a field is wrong — only
  that it differs from the reference.
- It is not a substitute for human review on high-stakes outputs. It narrows
  the set of records a human needs to look at by surfacing the failures.
