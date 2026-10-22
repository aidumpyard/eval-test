"""
dev_check.py
------------
AI Factory — Development smoke test utility.

Purpose: fast, local regression check during prompt development.
         Catches output drift BEFORE it reaches production.

No API key. No model download. Runs in milliseconds.

Install:
    pip install rouge-score

Usage:
    from dev_check import smoke_test, SmokeResult

    result = smoke_test(
        reference="The customer account was suspended due to missed payments.",
        candidate="Account suspended: payment overdue.",
    )
    print(result.verdict)   # PASS / WARN / FAIL
    print(result.rouge_l)   # e.g. 0.62
    print(result.notes)     # list of human-readable observations
"""

from __future__ import annotations
from dataclasses import dataclass, field
from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# These are starting points. Calibrate against your own examples.
# A drop of 0.05 across 3+ examples in a batch = investigate.
WARN_THRESHOLD = 0.05   # relative drop from a baseline triggers WARN
FAIL_THRESHOLD = 0.15   # relative drop triggers FAIL

# Absolute floor: below this score the output is structurally different
ABSOLUTE_FLOOR = 0.30


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
@dataclass
class SmokeResult:
    rouge_l: float
    """ROUGE-L F1 score. Range [0, 1]. Higher = more overlap."""

    verdict: str
    """PASS / WARN / FAIL — relative to baseline if provided, else absolute."""

    notes: list[str] = field(default_factory=list)
    """Human-readable observations about the result."""

    baseline: float | None = None
    """Baseline score this result was compared against, if provided."""

    delta: float | None = None
    """Score change from baseline (negative = regression)."""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def smoke_test(
    reference: str,
    candidate: str,
    baseline: float | None = None,
) -> SmokeResult:
    """
    Compare candidate output against a reference using ROUGE-L F1.

    Parameters
    ----------
    reference : str
        The known-good or expected output (your ground truth example).
    candidate : str
        The new output you want to check (from the latest prompt version).
    baseline : float | None
        A previously recorded ROUGE-L score for this example.
        If provided, verdict is relative (did we regress?).
        If None, verdict is absolute (is the score above the floor?).

    Returns
    -------
    SmokeResult
        .rouge_l   — float score
        .verdict   — PASS / WARN / FAIL
        .notes     — list of observations
        .baseline  — baseline score if provided
        .delta     — score change from baseline if provided
    """
    reference = reference.strip()
    candidate = candidate.strip()
    if not reference or not candidate:
        raise ValueError("Both reference and candidate must be non-empty.")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    score = round(scores["rougeL"].fmeasure, 4)

    notes: list[str] = []
    delta: float | None = None
    verdict: str

    if baseline is not None:
        delta = round(score - baseline, 4)
        drop = baseline - score  # positive = regression

        if drop >= FAIL_THRESHOLD:
            verdict = "FAIL"
            notes.append(
                f"Score dropped {drop:.3f} from baseline {baseline:.3f}. "
                "Significant regression — review output before proceeding."
            )
        elif drop >= WARN_THRESHOLD:
            verdict = "WARN"
            notes.append(
                f"Score dropped {drop:.3f} from baseline {baseline:.3f}. "
                "Minor regression — check a few outputs manually."
            )
        else:
            verdict = "PASS"
            if delta > 0.05:
                notes.append(f"Score improved by {delta:.3f} from baseline.")
    else:
        # Absolute mode
        if score < ABSOLUTE_FLOOR:
            verdict = "FAIL"
            notes.append(
                f"Score {score:.3f} is below floor {ABSOLUTE_FLOOR}. "
                "Output is structurally different from reference."
            )
        elif score < ABSOLUTE_FLOOR + 0.15:
            verdict = "WARN"
            notes.append(
                f"Score {score:.3f} is low. Output may have drifted. "
                "Set a baseline from a known-good run for relative tracking."
            )
        else:
            verdict = "PASS"

    # Length signal — catches outputs that got too long or too short
    ref_words = len(reference.split())
    cand_words = len(candidate.split())
    length_ratio = cand_words / ref_words if ref_words > 0 else 0

    if length_ratio < 0.5:
        notes.append(
            f"Candidate is much shorter than reference ({cand_words} vs {ref_words} words). "
            "May be truncated or under-generating."
        )
    elif length_ratio > 2.0:
        notes.append(
            f"Candidate is much longer than reference ({cand_words} vs {ref_words} words). "
            "May be over-generating or hallucinating extra content."
        )

    return SmokeResult(
        rouge_l=score,
        verdict=verdict,
        notes=notes,
        baseline=baseline,
        delta=delta,
    )


# ---------------------------------------------------------------------------
# Batch runner — use this for your smoke-test suite
# ---------------------------------------------------------------------------
def run_suite(
    examples: list[dict],
    fail_threshold: int = 3,
) -> dict:
    """
    Run smoke_test over a list of examples and return a summary.

    Parameters
    ----------
    examples : list of dicts, each with keys:
        - "reference": str
        - "candidate": str
        - "baseline": float (optional)
        - "label": str (optional, for display)
    fail_threshold : int
        Number of WARN/FAIL results that triggers an overall FAIL verdict.

    Returns
    -------
    dict with keys:
        - "overall": PASS / FAIL
        - "results": list of SmokeResult
        - "summary": human-readable string
    """
    results = []
    for ex in examples:
        r = smoke_test(
            reference=ex["reference"],
            candidate=ex["candidate"],
            baseline=ex.get("baseline"),
        )
        r._label = ex.get("label", f"example_{len(results)+1}")
        results.append(r)

    fails = sum(1 for r in results if r.verdict in ("WARN", "FAIL"))
    overall = "FAIL" if fails >= fail_threshold else "PASS"
    summary = (
        f"{overall} — {fails}/{len(results)} examples flagged "
        f"(threshold: {fail_threshold})"
    )
    return {"overall": overall, "results": results, "summary": summary}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        {
            "label": "paraphrase",
            "reference": "The customer account was suspended due to missed payments over three consecutive months.",
            "candidate": "Account suspended after three months of missed payments.",
            "baseline": 0.55,
        },
        {
            "label": "on-topic but shorter",
            "reference": "Please review the attached risk assessment report and provide your sign-off by end of day Friday.",
            "candidate": "Review risk report and sign off by Friday.",
            "baseline": 0.50,
        },
        {
            "label": "off-topic regression",
            "reference": "The transaction was flagged for manual review due to unusual activity.",
            "candidate": "The weather forecast shows rain tomorrow in the city centre.",
        },
    ]

    suite = run_suite(examples, fail_threshold=2)
    print(f"\nOverall: {suite['overall']}")
    print(f"Summary: {suite['summary']}\n")
    for r in suite["results"]:
        print(f"  [{r.verdict}] {r._label} — ROUGE-L: {r.rouge_l}")
        for note in r.notes:
            print(f"         ⚠  {note}")
