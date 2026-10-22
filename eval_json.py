"""
eval_json.py
------------
AI Factory — JSON output evaluator (adapter layer).

Your tool calls an LLM and extracts a NESTED JSON object per input. This
module scores chosen fields of that JSON against expected (golden) references,
reusing the existing metric engines and adding exact match:

  - rouge      ROUGE-L word overlap  (fast, no model)   -> dev_check.smoke_test
  - bertscore  semantic F1           (needs torch)       -> prod_monitor.score_outputs
  - exact      normalized equality   (right/wrong)       -> built in here

Each field can use its OWN metric, because different field types need
different tests:
    free text (summary)        -> bertscore (meaning, not wording)
    category  (decision/level) -> exact     (right or wrong, no model)
    tag list  (risks)          -> rouge or exact

The engines take flat (reference, candidate) string pairs. This adapter does
the three things that stand between your JSON and those engines:
    1. pair candidate records with reference records (by id, else by position)
    2. pull each requested field out of the nested JSON via a dotted path
    3. coerce the field value to a string (lists are joined, dicts dumped)

Scores from different fields are NOT comparable and are never pooled.

Install:
    pip install rouge-score      # for ROUGE-L (always)
    pip install bert-score       # for BERTScore (optional, pulls torch ~700 MB)

Python usage:
    from eval_json import evaluate, load_records

    report = evaluate(
        candidates=load_records("outputs.json"),
        references=load_records("golden.json"),
        fields=[
            {"path": "summary",                  "metric": "bertscore"},
            {"path": "risk_assessment.decision", "metric": "exact"},
            {"path": "risks", "list_sep": "; ",  "metric": "rouge"},
        ],
        key="id",            # pair records by this identifier
        metric="rouge",      # default metric for fields that don't set one
    )
    report.print_summary()

Command line (score several output files against one reference):
    python3 eval_json.py --ref golden.json --out run_v1.json run_v2.json \
        --fields summary:bertscore risk_assessment.decision:exact risks:rouge
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from dev_check import smoke_test

# BERTScore engine is optional — only used if a field asks for it.
# prod_monitor imports even without torch (it guards its own import), so we
# also read its _BERT_OK flag to know whether BERTScore can actually run.
try:
    from prod_monitor import score_outputs, DriftReport, _BERT_OK as _BERT_AVAILABLE
except Exception:  # noqa: BLE001 — prod_monitor itself failed to import
    _BERT_AVAILABLE = False

VALID_METRICS = ("rouge", "bertscore", "exact", "both")


# ---------------------------------------------------------------------------
# Loading records
# ---------------------------------------------------------------------------
def load_records(path: str) -> list[dict]:
    """
    Load records from a .json file (a top-level array, or a single object)
    or a .jsonl file (one JSON object per line). Returns a list of dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(
        f"{path}: expected a JSON array or object, got {type(data).__name__}."
    )


# ---------------------------------------------------------------------------
# Field extraction from nested JSON
# ---------------------------------------------------------------------------
def extract_text(record: dict, path: str, list_sep: str = "\n") -> str | None:
    """
    Pull a value from a (possibly nested) record using a dotted path and
    coerce it to a string suitable for text metrics.

    Path syntax
    -----------
    "summary"                    -> record["summary"]
    "risk_assessment.decision"   -> record["risk_assessment"]["decision"]
    "risks.0"                    -> record["risks"][0]   (numeric = list index)

    Coercion
    --------
    str -> stripped; num/bool -> str(); list -> joined with `list_sep`;
    dict -> compact sorted JSON; None / missing -> None.
    """
    cur: Any = record
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return _to_text(cur, list_sep)


def _to_text(value: Any, list_sep: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_to_text(v, list_sep) or "" for v in value]
        return list_sep.join(p for p in parts if p)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


# ---------------------------------------------------------------------------
# Pairing candidate records with reference records
# ---------------------------------------------------------------------------
def _pair_records(
    candidates: list[dict],
    references: list[dict],
    key: str | None,
) -> list[tuple[str, dict, dict]]:
    """
    Return (label, candidate_record, reference_record) triples.

    With `key`: pair by that identifier (safe when order/length differ).
    Without `key`: pair positionally and require equal length.
    """
    if key:
        ref_by_key = {r[key]: r for r in references if key in r}
        paired, missing = [], []
        for c in candidates:
            if key not in c:
                missing.append("<no key>")
                continue
            k = c[key]
            if k in ref_by_key:
                paired.append((str(k), c, ref_by_key[k]))
            else:
                missing.append(str(k))
        if missing:
            print(f"  (!) {len(missing)} candidate(s) had no matching reference "
                  f"by '{key}': {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        return paired

    if len(candidates) != len(references):
        raise ValueError(
            f"Positional pairing needs equal lengths "
            f"(candidates={len(candidates)}, references={len(references)}). "
            f"Pass key='id' to pair by an identifier instead."
        )
    return [(f"row_{i}", c, r) for i, (c, r) in enumerate(zip(candidates, references))]


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------
@dataclass
class RecordScore:
    """One (reference, candidate) comparison for one field."""
    label: str
    score: float
    verdict: str = ""      # PASS/WARN/FAIL for rouge & exact; "" for bertscore
    note: str = ""


@dataclass
class FieldEval:
    field_path: str
    metric: str                               # rouge | bertscore | exact
    n_pairs: int
    mean_score: float
    per_record: list[RecordScore] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)   # labels missing the field
    overall_verdict: str = ""                 # bertscore drift verdict
    notes: list[str] = field(default_factory=list)      # extra observations


@dataclass
class EvalReport:
    field_evals: list[FieldEval] = field(default_factory=list)

    def mean_for(self, field_path: str, metric: str) -> float | None:
        for fe in self.field_evals:
            if fe.field_path == field_path and fe.metric == metric:
                return fe.mean_score
        return None

    def print_summary(self) -> None:
        print("\n=== JSON eval summary ===")
        for fe in self.field_evals:
            head = (f"\n  Field: {fe.field_path}  [{fe.metric}]  "
                    f"n={fe.n_pairs}  mean={fe.mean_score:.4f}")
            if fe.overall_verdict:
                head += f"  verdict={fe.overall_verdict}"
            print(head)

            if fe.skipped:
                print(f"    skipped (field missing): {len(fe.skipped)} "
                      f"-> {fe.skipped[:5]}{' ...' if len(fe.skipped) > 5 else ''}")

            flagged = [r for r in fe.per_record if r.verdict in ("WARN", "FAIL")]
            if fe.per_record and fe.metric in ("rouge", "exact"):
                print(f"    verdicts: "
                      f"{sum(r.verdict == 'PASS' for r in fe.per_record)} PASS / "
                      f"{sum(r.verdict == 'WARN' for r in fe.per_record)} WARN / "
                      f"{sum(r.verdict == 'FAIL' for r in fe.per_record)} FAIL")
            for r in flagged[:5]:
                line = f"      [{r.verdict}] {r.label}  score={r.score:.4f}"
                if r.note:
                    line += f"  — {r.note}"
                print(line)
            for note in fe.notes:
                print(f"      - {note}")


# ---------------------------------------------------------------------------
# Per-metric scorers
# ---------------------------------------------------------------------------
def _normalize(s: str) -> str:
    """Casefold + collapse whitespace, for exact match of categorical values."""
    return " ".join(s.split()).casefold()


def _score_exact(path, labels, refs, cands) -> FieldEval:
    per = []
    for lbl, ref, cand in zip(labels, refs, cands):
        hit = _normalize(ref) == _normalize(cand)
        per.append(RecordScore(
            label=lbl,
            score=1.0 if hit else 0.0,
            verdict="PASS" if hit else "FAIL",
            note="" if hit else f"expected {ref!r}, got {cand!r}",
        ))
    mean = round(sum(r.score for r in per) / len(per), 4)
    return FieldEval(path, "exact", len(per), mean, per_record=per)


def _score_rouge(path, labels, refs, cands, baseline) -> FieldEval:
    per = []
    for lbl, ref, cand in zip(labels, refs, cands):
        r = smoke_test(reference=ref, candidate=cand, baseline=baseline)
        per.append(RecordScore(
            label=lbl,
            score=r.rouge_l,
            verdict=r.verdict,
            note=r.notes[0] if r.notes else "",
        ))
    mean = round(sum(r.score for r in per) / len(per), 4)
    return FieldEval(path, "rouge", len(per), mean, per_record=per)


def _score_bertscore(path, labels, refs, cands, baseline) -> FieldEval:
    drift: DriftReport = score_outputs(
        references=refs, candidates=cands, baseline_mean_f1=baseline
    )
    per = [RecordScore(label=lbl, score=f1)
           for lbl, f1 in zip(labels, drift.per_pair_f1)]
    return FieldEval(
        path, "bertscore", len(per), drift.mean_f1,
        per_record=per, overall_verdict=drift.verdict, notes=drift.notes,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def evaluate(
    candidates: list[dict],
    references: list[dict],
    fields: list,
    key: str | None = "id",
    metric: str = "rouge",
) -> EvalReport:
    """
    Score chosen fields of candidate JSON records against reference records.

    Parameters
    ----------
    candidates : list of dicts — your tool's JSON output, one per input.
    references : list of dicts — expected/golden outputs, same schema.
    fields     : list of field specs. Each item is either:
                   - a str dotted path:  "risk_assessment.decision"
                   - a dict:
                       {"path": "summary",
                        "metric": "bertscore",  # per-field; defaults to `metric`
                        "list_sep": "; ",        # how to join list values
                        "baseline": 0.74}        # optional, rouge/bertscore only
    key        : identifier field to pair records by. None = pair by position.
    metric     : default metric for fields that don't set their own.
                 "rouge" | "bertscore" | "exact" | "both".

    Returns
    -------
    EvalReport — call .print_summary(), or read .field_evals for raw data.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}.")

    # Resolve each field's metric up front so we can validate before any work.
    resolved = []
    for spec in fields:
        if isinstance(spec, str):
            spec = {"path": spec}
        m = spec.get("metric", metric)
        if m not in VALID_METRICS:
            raise ValueError(f"field {spec['path']!r}: bad metric {m!r}.")
        resolved.append((spec, m))

    needs_bert = any(m in ("bertscore", "both") for _, m in resolved)
    if needs_bert and not _BERT_AVAILABLE:
        raise ImportError(
            "A field requested BERTScore but prod_monitor/bert-score is "
            "unavailable.\nRun: pip install bert-score   (pulls torch ~700 MB)."
        )

    pairs = _pair_records(candidates, references, key)
    report = EvalReport()

    for spec, m in resolved:
        path = spec["path"]
        list_sep = spec.get("list_sep", "\n")
        baseline = spec.get("baseline")

        labels, refs, cands, skipped = [], [], [], []
        for label, cand_rec, ref_rec in pairs:
            ref_txt = extract_text(ref_rec, path, list_sep)
            cand_txt = extract_text(cand_rec, path, list_sep)
            if not ref_txt or not cand_txt:
                skipped.append(label)
                continue
            labels.append(label)
            refs.append(ref_txt)
            cands.append(cand_txt)

        wanted = ["rouge", "bertscore"] if m == "both" else [m]
        if not labels:
            for w in wanted:
                report.field_evals.append(FieldEval(path, w, 0, 0.0, skipped=skipped))
            continue

        for w in wanted:
            if w == "exact":
                fe = _score_exact(path, labels, refs, cands)
            elif w == "rouge":
                fe = _score_rouge(path, labels, refs, cands, baseline)
            else:  # bertscore
                fe = _score_bertscore(path, labels, refs, cands, baseline)
            fe.skipped = skipped
            report.field_evals.append(fe)

    return report


# ---------------------------------------------------------------------------
# Compare several output files against one reference file
# ---------------------------------------------------------------------------
def compare_files(
    ref_path: str,
    out_paths: list[str],
    fields: list,
    key: str | None = "id",
    metric: str = "rouge",
) -> dict[str, EvalReport]:
    """
    Score each output file against the SAME reference file and print a
    side-by-side table of mean score per (field, metric) per file.

    Returns {output_file_path: EvalReport} for further inspection.
    """
    references = load_records(ref_path)
    reports: dict[str, EvalReport] = {}
    for out_path in out_paths:
        print(f"\n--- Scoring {out_path} vs {ref_path} ---")
        reports[out_path] = evaluate(
            candidates=load_records(out_path),
            references=references,
            fields=fields,
            key=key,
            metric=metric,
        )

    # Build the comparison table: rows = (field, metric), cols = files.
    row_keys: list[tuple[str, str]] = []
    for rep in reports.values():
        for fe in rep.field_evals:
            if (fe.field_path, fe.metric) not in row_keys:
                row_keys.append((fe.field_path, fe.metric))

    print("\n=== Comparison (mean score) ===")
    names = [p.split("/")[-1] for p in out_paths]
    name_w = max([len("field [metric]")] +
                 [len(f"{p} [{m}]") for p, m in row_keys])
    col_w = max(12, *(len(n) for n in names))
    print("  " + "field [metric]".ljust(name_w) + "  " +
          "  ".join(n.rjust(col_w) for n in names))
    print("  " + "-" * (name_w + 2 + (col_w + 2) * len(names) - 2))
    for p, m in row_keys:
        cells = []
        for out_path in out_paths:
            val = reports[out_path].mean_for(p, m)
            cells.append((f"{val:.4f}" if val is not None else "-").rjust(col_w))
        print("  " + f"{p} [{m}]".ljust(name_w) + "  " + "  ".join(cells))

    return reports


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def _parse_field_arg(arg: str) -> dict:
    """
    CLI field syntax:  path            -> use the default --metric
                       path:metric     -> per-field metric (e.g. summary:bertscore)
    Dotted paths use '.', so ':' is unambiguous as the metric separator.
    """
    if ":" in arg:
        path, m = arg.rsplit(":", 1)
        if m in VALID_METRICS:
            return {"path": path, "metric": m}
    return {"path": arg}


def _main_cli(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Score one or more output JSON files against a reference JSON file.",
    )
    parser.add_argument("--ref", required=True,
                        help="Reference (golden) JSON/JSONL file.")
    parser.add_argument("--out", required=True, nargs="+",
                        help="One or more output JSON/JSONL files to score.")
    parser.add_argument("--fields", required=True, nargs="+",
                        help="Fields to score. Use path or path:metric, e.g. "
                             "summary:bertscore risk_assessment.decision:exact")
    parser.add_argument("--key", default="id",
                        help="Record identifier to pair on (default: id). "
                             "Use --key '' to pair by position.")
    parser.add_argument("--metric", default="rouge",
                        choices=["rouge", "bertscore", "exact", "both"],
                        help="Default metric for fields with no per-field metric.")
    args = parser.parse_args(argv)

    compare_files(
        ref_path=args.ref,
        out_paths=args.out,
        fields=[_parse_field_arg(f) for f in args.fields],
        key=args.key or None,
        metric=args.metric,
    )


# ---------------------------------------------------------------------------
# Entry point — CLI if args given, else the built-in ROUGE-L + exact demo
# ---------------------------------------------------------------------------
if __name__ == "__main__" and len(__import__("sys").argv) > 1:
    _main_cli(__import__("sys").argv[1:])

elif __name__ == "__main__":
    candidates = [
        {
            "id": "case-001",
            "summary": "Account suspended after three months of missed payments.",
            "risk_assessment": {"decision": "suspend", "level": "high"},
            "risks": ["late payment", "credit exposure"],
        },
        {
            "id": "case-002",
            "summary": "The account was closed permanently at the user's request.",
            "risk_assessment": {"decision": "close", "level": "high"},
            "risks": ["user request"],
        },
    ]
    references = [
        {
            "id": "case-001",
            "summary": "The customer account was suspended due to missed payments over three months.",
            "risk_assessment": {"decision": "suspend", "level": "high"},
            "risks": ["late payment", "credit exposure", "operational"],
        },
        {
            "id": "case-002",
            "summary": "Identity confirmed; the account has been reinstated.",
            "risk_assessment": {"decision": "reinstate", "level": "low"},
            "risks": ["identity check pending"],
        },
    ]

    report = evaluate(
        candidates=candidates,
        references=references,
        fields=[
            {"path": "summary", "metric": "rouge"},            # swap to bertscore in prod
            {"path": "risk_assessment.decision", "metric": "exact"},
            {"path": "risk_assessment.level", "metric": "exact"},
            {"path": "risks", "list_sep": "; ", "metric": "rouge"},
        ],
        key="id",
    )
    report.print_summary()
