"""
prod_monitor.py
---------------
AI Factory — Production monitoring utility.

Two responsibilities:
  1. HASH LOGGING   — deterministic audit trail for every inference call.
                      Lets you reproduce exactly what the system produced at time T.
  2. BERTSCORE      — semantic drift detection across your eval set.
                      Catches model/prompt degradation over time.

Install:
    pip install bert-score
    # bert-score pulls torch (~700 MB). One-time cost.

Usage — hash logging:
    from prod_monitor import log_inference, verify_inference

    record = log_inference(
        model="gpt-4o",
        model_version="2024-11-20",
        temperature=0.0,
        input_text="Summarise the risk report.",
        output_text="The report identifies three key risks...",
    )
    # record is a dict — store it in your logging system (DB, S3, etc.)

Usage — BERTScore:
    from prod_monitor import score_outputs, DriftReport

    report = score_outputs(
        references=["Expected output A", "Expected output B"],
        candidates=["Actual output A", "Actual output B"],
    )
    print(report.mean_f1)       # e.g. 0.87
    print(report.verdict)       # STABLE / DEGRADED
    print(report.notes)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# BERTScore is an optional import — hash logging works without it
try:
    from bert_score import score as bert_score_fn
    _BERT_OK = True
except ImportError:
    _BERT_OK = False


# ---------------------------------------------------------------------------
# Thresholds for drift detection
# ---------------------------------------------------------------------------
DEGRADATION_THRESHOLD = 0.03   # mean F1 drop vs baseline triggers DEGRADED
WARN_THRESHOLD        = 0.01   # smaller drop triggers WARN


# ---------------------------------------------------------------------------
# 1. HASH LOGGING
# ---------------------------------------------------------------------------

def log_inference(
    model: str,
    model_version: str,
    temperature: float,
    input_text: str,
    output_text: str,
    extra_params: dict | None = None,
) -> dict[str, Any]:
    """
    Create a deterministic audit record for one inference call.

    How it works
    ------------
    SHA-256 hashes the input and output independently.
    The input_hash + model + version + temperature together form a
    'reproduction key' — if you replay these exact parameters you should
    get the same output (at temperature=0). The output_hash proves you did.

    Parameters
    ----------
    model         : model name, e.g. "gpt-4o"
    model_version : pinned version string, e.g. "2024-11-20"
                    NEVER use "latest" in production — it silently changes.
    temperature   : sampler temperature used for this call
    input_text    : the exact prompt sent to the model
    output_text   : the exact output received
    extra_params  : any other parameters (max_tokens, top_p, etc.)

    Returns
    -------
    dict — store this record in your logging system.
    Fields: timestamp, model, model_version, temperature,
            input_hash, output_hash, input_length_words,
            output_length_words, extra_params
    """
    input_hash  = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    output_hash = hashlib.sha256(output_text.encode("utf-8")).hexdigest()

    record = {
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "model":               model,
        "model_version":       model_version,
        "temperature":         temperature,
        "input_hash":          input_hash,
        "output_hash":         output_hash,
        "input_length_words":  len(input_text.split()),
        "output_length_words": len(output_text.split()),
        "extra_params":        extra_params or {},
    }
    return record


def verify_inference(
    record: dict[str, Any],
    input_text: str,
    output_text: str,
) -> dict[str, bool]:
    """
    Verify that a stored record matches the provided texts.

    Use this to prove that a stored audit record corresponds to
    specific input/output texts — e.g. during incident investigation.

    Returns
    -------
    dict with keys:
        "input_match"  : bool — input hash matches
        "output_match" : bool — output hash matches
        "both_match"   : bool — shorthand for both
    """
    input_hash  = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    output_hash = hashlib.sha256(output_text.encode("utf-8")).hexdigest()

    input_match  = input_hash  == record.get("input_hash")
    output_match = output_hash == record.get("output_hash")
    return {
        "input_match":  input_match,
        "output_match": output_match,
        "both_match":   input_match and output_match,
    }


# ---------------------------------------------------------------------------
# 2. BERTSCORE — semantic drift detection
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    mean_f1: float
    """Mean BERTScore F1 across all pairs. Higher = more similar."""

    std_f1: float
    """Standard deviation. High std means inconsistent outputs."""

    verdict: str
    """STABLE / WARN / DEGRADED — relative to baseline if provided."""

    notes: list[str] = field(default_factory=list)
    """Human-readable observations."""

    baseline_mean_f1: float | None = None
    delta: float | None = None
    per_pair_f1: list[float] = field(default_factory=list)
    model_used: str = ""


def score_outputs(
    references: list[str],
    candidates: list[str],
    baseline_mean_f1: float | None = None,
    model_type: str = "distilbert-base-uncased",
    lang: str = "en",
) -> DriftReport:
    """
    Compute BERTScore F1 between reference and candidate outputs.

    When to use
    -----------
    Run this on your eval set after every deploy or on a daily schedule.
    Track mean_f1 over time. A downward trend across 3+ consecutive runs
    is your signal to investigate — not a single data point.

    Parameters
    ----------
    references          : list of reference (expected) outputs
    candidates          : list of candidate (actual) outputs
    baseline_mean_f1    : mean F1 from a previous known-good run.
                          If provided, verdict is relative (did we regress?).
    model_type          : BERTScore model. distilbert is fast and sufficient
                          for drift detection. Use 'roberta-large' for
                          higher precision at higher cost.
    lang                : language of the texts

    Returns
    -------
    DriftReport

    Notes
    -----
    BERTScore does token-level matching, not document-level embedding.
    It works correctly on long texts because it matches tokens across
    the full text — unlike cosine similarity which truncates at ~256 tokens.
    """
    if not _BERT_OK:
        raise ImportError(
            "bert-score is not installed.\n"
            "Run: pip install bert-score\n"
            "Note: this pulls torch (~700 MB) on first install."
        )
    if len(references) != len(candidates):
        raise ValueError("references and candidates must be the same length.")
    if not references:
        raise ValueError("At least one reference/candidate pair is required.")

    _, _, F1 = bert_score_fn(
        candidates,
        references,
        model_type=model_type,
        lang=lang,
        verbose=False,
    )

    f1_list  = [round(float(f), 4) for f in F1]
    mean_f1  = round(sum(f1_list) / len(f1_list), 4)
    variance = sum((f - mean_f1) ** 2 for f in f1_list) / len(f1_list)
    std_f1   = round(variance ** 0.5, 4)

    notes: list[str] = []
    delta: float | None = None
    verdict: str

    if baseline_mean_f1 is not None:
        delta = round(mean_f1 - baseline_mean_f1, 4)
        drop  = baseline_mean_f1 - mean_f1

        if drop >= DEGRADATION_THRESHOLD:
            verdict = "DEGRADED"
            notes.append(
                f"Mean F1 dropped {drop:.4f} from baseline {baseline_mean_f1:.4f}. "
                "Semantic quality has degraded — investigate before next deploy."
            )
        elif drop >= WARN_THRESHOLD:
            verdict = "WARN"
            notes.append(
                f"Mean F1 dropped {drop:.4f} from baseline. "
                "Small regression — monitor over next 2–3 runs before acting."
            )
        else:
            verdict = "STABLE"
    else:
        # No baseline — absolute mode
        if mean_f1 >= 0.85:
            verdict = "STABLE"
        elif mean_f1 >= 0.70:
            verdict = "WARN"
            notes.append(
                f"Mean F1 {mean_f1:.4f} is moderate. Establish a baseline from a "
                "known-good run so future comparisons are relative, not absolute."
            )
        else:
            verdict = "DEGRADED"
            notes.append(
                f"Mean F1 {mean_f1:.4f} is low. Outputs differ substantially from "
                "references. Review the lowest-scoring pairs first."
            )

    if std_f1 > 0.10:
        notes.append(
            f"High score variance (std={std_f1:.4f}). Some outputs are much worse "
            "than others. Check the lowest-scoring pairs individually."
        )

    # Surface the worst pairs
    if len(f1_list) > 1:
        min_score = min(f1_list)
        min_idx   = f1_list.index(min_score)
        if min_score < mean_f1 - 0.10:
            notes.append(
                f"Lowest-scoring pair: index {min_idx} (F1={min_score:.4f}). "
                "Inspect this output first."
            )

    return DriftReport(
        mean_f1=mean_f1,
        std_f1=std_f1,
        verdict=verdict,
        notes=notes,
        baseline_mean_f1=baseline_mean_f1,
        delta=delta,
        per_pair_f1=f1_list,
        model_used=model_type,
    )


# ---------------------------------------------------------------------------
# Self-test (hash logging only — BERTScore requires torch)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Hash logging test ===")
    record = log_inference(
        model="gpt-4o",
        model_version="2024-11-20",
        temperature=0.0,
        input_text="Summarise the Q3 risk report in two sentences.",
        output_text="The Q3 risk report identifies three critical risks: liquidity, credit exposure, and operational resilience. Immediate action is recommended on liquidity.",
    )
    print(json.dumps(record, indent=2))

    print("\n=== Verify matching texts ===")
    result = verify_inference(
        record=record,
        input_text="Summarise the Q3 risk report in two sentences.",
        output_text="The Q3 risk report identifies three critical risks: liquidity, credit exposure, and operational resilience. Immediate action is recommended on liquidity.",
    )
    print(result)

    print("\n=== Verify tampered output ===")
    result_tampered = verify_inference(
        record=record,
        input_text="Summarise the Q3 risk report in two sentences.",
        output_text="The Q3 risk report looks fine. No action needed.",
    )
    print(result_tampered)
