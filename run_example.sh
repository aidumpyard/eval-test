#!/usr/bin/env bash
# run_example.sh — score the example output files against the reference.
# Pure offline eval: no LLM, no network. Run from this directory:  ./run_example.sh
set -euo pipefail
cd "$(dirname "$0")"

python3 eval_json.py \
  --ref    reference.json \
  --out    output_a.json output_b.json \
  --fields assessment.decision:exact \
           assessment.risk_level:exact \
           assessment.summary:rouge \
           factors:rouge \
           recommended_action.primary:rouge \
           recommended_action.alternative:rouge \
           follow_up.owner:exact \
           follow_up.due_in_days:exact \
  --key    id
