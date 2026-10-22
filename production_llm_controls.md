# Production Controls for LLM and Agentic Systems

**Purpose:** detect, explain and respond to changes in what an LLM system produces, whether the change comes from a model swap, from the provider's serving infrastructure, from text an invisible layer adds (provider, gateway or safety filter), or from your own prompts, tools and data.

**Status:** this file supersedes the parts of `semantic_similarity_guide.md`, `prod_monitor.py` and `dev_check.py` listed in §8. Hash logging and version pinning from those files are kept.

**How to read this:** §1–§3 are the operational front: what can change, which control covers it, and in what order to build. §4–§7 explain how to build each control. Every threshold in this file is either derived by a stated method or computed; none is a rule of thumb.

---

## 1. What can change without you knowing

| # | Source of change | Example | Visible in the API response? |
|---|---|---|---|
| S1 | Model weights | Alias moves to a new snapshot; snapshot retired | Sometimes (`model` field) |
| S2 | Serving infrastructure, same model | Routing, hardware or compiler bugs that degrade output. Anthropic's Sept 2025 postmortem describes three such bugs that affected pinned models ([source](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues)) | No |
| S3 | Hidden input text | Provider-side system text, gateway or proxy templates, a safety layer that rewrites prompts | Indirectly (input token count) |
| S4 | Hidden output processing | Filters, truncation, rewriting | Indirectly (stop reason, length, format) |
| S5 | Your own changes | Prompt edits, tool schemas, parameters, code | Yes, if versioned |
| S6 | Environment | Data returned by tools or APIs, retrieval index contents, time-dependent facts | No |
| S7 | Your measuring instrument | Judge or embedding model behind an alias changes | No |

Two facts shape the whole design:
- **Pinning a model does not freeze behaviour** (S2). Some form of behavioural canary is mandatory.
- **A detected change is not necessarily a regression.** "Different" and "worse" are separate questions with separate controls (D3 versus D5).

---

## 2. Control catalogue

Types: **P** = preventive, **D** = detective, **C** = corrective.

| ID | Control | Covers | Frequency | Signal | Skip when |
|---|---|---|---|---|---|
| P1 | Configuration manifest: pin and version everything | S1, S5 | Every deploy | Manifest hash | Never |
| P2 | Change gate: paired regression test before any change | S5 | Every change | Paired test on golden set + D3 tests | Never |
| P3 | Prompt qualification: stability test before going live | Intrinsic variance | Once per new or changed prompt | Modal share / semantic entropy | Output is fully validated by deterministic code downstream |
| P4 | Runtime output validators | S4, any | Every call | Schema / invariant pass rate | Never for structured outputs |
| P5 | Runtime self-consistency on high-stakes calls | Intrinsic variance | Per selected call | Agreement across n samples | Unless P(error caught) × cost of error > (n−1) × cost per call |
| D1 | Provenance log + model identity check | S1, audit | Every call | `model` / fingerprint change | Never |
| D2 | Input-token offset check | S3 | Every canary run | Any change in tokens for an identical request | Never (cost ≈ 0) |
| D3 | Canary suite with two-sample tests | S1–S4 | Scheduled + every deploy | Structural, meaning-group and distribution tests vs baseline | Never; drop only the semantic layer when outputs are fully structured |
| D4 | Production traffic monitors | S1–S6, real impact | Continuous | Control charts on structural features | Below the traffic needed for a stable baseline (§6.6) |
| D5 | Quality check on golden set | "Worse", not just "different" | Scheduled + on any D-alert | Graded pass rate | Never |
| D6 | Agent reliability (pass^k) with record/replay | S1–S6 for agents | Scheduled | pass^k vs baseline | No multi-step tool use |
| D7 | Instrument integrity | S7 | Every monitoring run | Fixed pairs must score identically | Never |
| D8 | Provider notice watch | S1, S2 | Weekly | Deprecations, incidents, postmortems | Never |
| C1 | Attribution runbook | All | On alert | Decision tree (§5) | — |
| C2 | Fallback, rollback, kill switch | All | On confirmed regression | Switch time | — |
| C3 | Controlled re-baselining | All | On accepted change | Signed-off baseline version | — |

### Control record template (fill one in for each control)

```
Control ID / name:
Objective:                   what risk it addresses (from §1)
Type:                        P / D / C
Owner:                       control owner (runs it)  |  reviewer (independent)
Trigger / frequency:         derived from tolerated detection delay (§6.7)
Method:                      section reference
Metric and alert rule:       statistic, null distribution, alpha / FDR level
Evidence retained:           what is stored, where, for how long
Response on alert:           C1 path, escalation, SLA
Design-effectiveness test:   fault injections it must detect (§7), last result
Operating-effectiveness:     evidence that runs happened and alerts were triaged
Known blind spots:           from §9
```

---

## 3. Build order

Build deterministic, zero-noise controls first, then statistical ones.

1. **Phase 1 — deterministic:** P1, D1, D2, P4, D8. No statistics, no false positives.
   *Exit criterion:* every production call is logged with the manifest hash, and D2 runs on a schedule.
2. **Phase 2 — baseline:** build the probe set (§6.1) and collect the multi-day baseline (§6.2). Turn on the structural layer of D3, and D4.
   *Exit criterion:* the A/A false-alarm rate has been measured and is at or below the target α.
3. **Phase 3 — semantic and quality:** meaning-group and distribution layers of D3, D5, the P2 gate, and D7.
   *Exit criterion:* the fault-injection suite (§7) passes.
4. **Phase 4 — agents and high stakes:** D6, P3, P5.
   *Exit criterion:* attribution (§5) can tell model change from environment change on an injected fault.

---

## 4. Control methods

### P1 — Configuration manifest
- Record per deployment:
  - model snapshot ID (never an alias);
  - sampling parameters;
  - SHA-256 of the system prompt and each template;
  - tool schema versions;
  - retrieval index version;
  - gateway/proxy version;
  - judge and embedding model IDs plus file hashes.
- The **manifest hash** goes on every log line (D1), so any output can be traced to the exact configuration that produced it.
- Track the provider's deprecation schedule for each snapshot in use (D8). A retirement date is a forced change and must go through P2.

### P2 — Change gate (paired)
- Run old and new configurations on the **same** golden items, with k samples each.
- Compare pass/fail per item with a **McNemar test**, or per-item pass rates with a **Wilcoxon signed-rank test**. A paired design needs far fewer items than comparing two unpaired groups, because items that both configurations pass or both fail cancel out.
- Also run the D3 tests between the old and new configurations. This tells you what changed, not only whether quality moved.
- Gate rule: no statistically significant quality drop at the chosen α, and every D3 change is explained in the change record.

### P3 — Prompt qualification (where dispersion metrics belong)
- Sample n outputs per qualification prompt at production parameters.
- Group outputs into meaning groups (§6.4).
- Compute:
  - **modal share** (fraction of outputs in the largest group) with a Wilson interval;
  - **semantic entropy** H = −Σ p(c) log p(c) ([Farquhar et al., 2024](https://www.nature.com/articles/s41586-024-07421-0)).
- Rule: for tasks that should have one answer, the **lower Wilson bound** of modal share must exceed 1 − (disagreement rate the business can tolerate).
  - Example: 95/100 in the modal group gives a 95% Wilson interval of 88.8%–97.8%. With a 10% tolerance the lower bound must exceed 90%: 88.8% fails. With a 12% tolerance it must exceed 88%: it passes. A 95% observed agreement is not enough evidence for a 10% tolerance at n = 100.
- Prompts that fail need redesign, for example constrained output or decomposition. A threshold should not be loosened to let them through.

### P4 — Runtime validators
Deterministic checks on every output:
- JSON schema;
- enum membership;
- numeric ranges;
- required sections present;
- cited IDs exist in the supplied context;
- `stop_reason` is not `max_tokens`.

The pass rate of each validator also feeds D4 as a monitored feature.

### P5 — Runtime self-consistency (selective)
- For calls that drive irreversible or high-cost actions, sample n times and act only if agreement meets the P3 rule. Otherwise escalate to a human.
- The skip condition in §2 is an ROI inequality. Compute it with your own error cost and call cost.

### D1 — Provenance log and identity check
- Log per call:
  - manifest hash;
  - request hash and response hash (from the existing `prod_monitor.log_inference`);
  - the `model` field returned (and `system_fingerprint` where provided);
  - full `usage` block;
  - `stop_reason`;
  - latency.
- Alert on any returned model ID or fingerprint that differs from the manifest.
- The response hash is **audit evidence, not a drift detector**. Even at temperature 0, outputs are not guaranteed identical across calls ([Thinking Machines, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)), so hash mismatches are expected noise.

### D2 — Input-token offset check
- For each probe, send a **byte-identical** request and record total input tokens. On Anthropic, total = `input_tokens` + `cache_creation_input_tokens` + `cache_read_input_tokens`.
- Token counting is deterministic for identical input under an unchanged tokenizer and serving template. So the baseline value per probe is exact, and **any change is an alert**. No statistics are needed.
- A non-zero gap between your own count and the reported count is **not** evidence of hidden text. Chat templates and tool definitions add tokens legitimately (Anthropic documents a tool-use system prompt overhead), and some tokenizers are not public. Only a **change** is a signal.
- Run the same probes both directly against the provider and through your gateway. A difference between the two paths isolates gateway-added text.

### D3 — Canary suite with two-sample tests
Compare the current canary run (B) against the baseline (A), probe by probe, in three layers:

| Layer | Features | Test |
|---|---|---|
| Structural | Output length; `stop_reason`; refusal flag (pinned classifier or regex); schema validity; counts of bullets and headings; disclaimer-phrase hits; tool-call sequence; latency | Continuous features: two-sample Kolmogorov–Smirnov. Categorical features: chi-square, or Fisher exact when counts are small |
| Meaning groups | Assign each B output to A's meaning groups (§6.4), adding a "new" group when nothing matches | Chi-square test of homogeneity on group counts. Report Jensen–Shannon divergence as the effect size |
| Whole distribution | Embeddings from the pinned local model (D7) | [MMD](https://jmlr.org/papers/v13/gretton12a.html) with an RBF kernel (median-heuristic bandwidth) and a permutation test, or a [classifier two-sample test](https://arxiv.org/abs/1610.06545): cross-validated logistic regression on embeddings + structural features, with a binomial test against 0.5 accuracy |

Two kinds of probe are needed (§6.1):
- **Tripwires**: near-zero baseline variance, so they are maximally sensitive to change.
- **Representative probes**: sampled from real traffic, so they cover what users actually do.

Control false alarms across the many probe × feature tests with Benjamini–Hochberg at level q (§6.3). The calibration procedure in §6.2 decides whether those p-values can be trusted.

For long-form outputs that feed decisions only: split outputs into atomic claims, then compare each claim's **support rate** (fraction of samples that entail it) between A and B with a two-proportion test. Skip this otherwise. Claim extraction is LLM-based and adds its own variance.

### D4 — Production traffic monitors
- Compute the structural features from D3 on **all** production traffic, at no extra inference cost. Add:
  - cost per task;
  - tool error rate;
  - user retry rate;
  - explicit feedback.
- Stratify by route or task type. A shift in what users ask otherwise looks like model drift.
- Use **EWMA or CUSUM control charts** for slow drift and Shewhart charts for step changes ([NIST e-Handbook, ch. 6](https://www.itl.nist.gov/div898/handbook/)). Set the control limits from a baseline window, and recompute them only under C3.
- D4 shows real-world impact but cannot attribute cause, because inputs vary. D3 attributes cause but uses synthetic inputs. You need both.

### D5 — Quality check
- Use a golden set with expected outcomes or rubrics, graded by deterministic checks where possible. Otherwise use a pinned judge that is validated against human labels: measure judge–human agreement on at least the items used for the threshold, and report it.
- Run on a schedule and on any D-alert. Pass-rate change is tested with a two-proportion test (unpaired, over time) or McNemar (paired, across configurations).
- Sample sizes come from §6.5. A golden set of 15–30 items detects only large drops.

### D6 — Agent reliability
- Metric: **pass^k**, the probability that all k independent trials succeed ([τ-bench](https://arxiv.org/abs/2406.12045)). With c successes in n trials, estimate it as C(c,k) / C(n,k).
  - Example: 90/100 successes gives pass^8 ≈ 0.42, whereas 0.9⁸ ≈ 0.43.
- Run each agent canary in two modes:
  - **Replay**: tool outputs are recorded once and replayed. Only the model and the hidden layers can vary.
  - **Live**: real tools.
- Replay changed → model or hidden-layer cause. Only live changed → environment cause (S6).

### D7 — Instrument integrity
- Embedding, NLI and judge models must be local and version-pinned, with file hashes stored in the manifest. LLM judges reached through an API are used for explaining alerts, never for triggering them.
- Keep a fixed set of instrument-canary pairs. Each monitoring run re-scores them, and any deviation beyond floating-point tolerance invalidates that run.

### D8 — Provider notice watch
Each week, review:
- provider model deprecation pages;
- status and incident history;
- engineering postmortems;
- release notes;
- gateway or proxy changelogs.

Any announced change triggers an unscheduled D3 + D5 run.

---

## 5. Corrective controls

### C1 — Attribution runbook (run in order; stop at the first positive)
1. **D1 identity changed?** → S1. Confirm with the provider and run P2-style comparison to quantify impact.
2. **D2 token offset changed?** → S3. Run the direct-versus-gateway comparison to locate the layer.
3. **D7 instrument changed?** → the alert is invalid. Fix the instrument and re-run.
4. **D6 replay mode changed?** → model or hidden layer (S1/S2/S3/S4). Re-run D3 through the direct path to rule the gateway in or out.
5. **Only live mode or D4 changed?** → S6 environment, or a change in the traffic mix. Check retrieval index version, tool API changes, and the input distribution.
6. **Nothing localises?** → open a provider ticket attaching the D3 evidence (A/B samples, test statistics, manifest).

Then run D5: a change that is not a regression moves to C3, and a regression moves to C2.

### C2 — Fallback, rollback, kill switch
- Have a pre-qualified fallback configuration that has passed P2 and P3.
- Measure the switch time in a drill. That time is part of the control.

### C3 — Re-baselining
- Accepting a change creates a new, versioned baseline. Old baselines are retained as evidence.
- Sign-off requires the D3 diff, the D5 result and the attribution conclusion.
- Never re-baseline silently. Doing so turns a detector into a normaliser.

---

## 6. Methods for building the controls

### 6.1 Building the probe set
- **Tripwires:** prompts whose baseline output varies little, such as:
  - exact-format outputs;
  - closed-form answers;
  - fixed tool paths;
  - boundary prompts near refusal or policy edges, where hidden safety layers act first;
  - a long-context prompt, where routing problems show first.

  Confirm low variance with P3 before including one.
- **Representative probes:** stratified samples of real production inputs, PII-scrubbed, covering every route. Refresh them under C3, never ad hoc.
- **Size:** comes from the detection target (§6.5), not from a convention. Effects that hit many probes in the same direction can be pooled. Effects confined to one route need that route's probes to be sized on their own.

### 6.2 Building the baseline and calibrating the alarm (A/A)
1. Run the full canary suite under the frozen manifest on D separate days, at different times of day. Sampling across time captures load and serving variance that a single batch cannot.
2. For each pair of baseline days, run every D3 test as if one day were A and the other B. There is no real change between them, so every alert is a false alarm.
3. Compare the observed false-alarm rate with the nominal α:
   - **At or below α:** the per-run permutation and asymptotic p-values can be trusted. Use BH at level q.
   - **Above α:** time-to-time variance is real and the textbook null is wrong. Use the empirical distribution of each statistic across baseline day pairs, pooled over probes, as the null, and alert when the current statistic exceeds its (1 − α) quantile.
4. D days give D(D−1)/2 pairs, and an empirical (1 − α) quantile needs roughly ≥ 1/α null values. Pool across probes, or extend D, until it does.

### 6.3 Statistical reference

| Need | Method |
|---|---|
| Interval on a proportion | Wilson score interval |
| Rate change, unpaired | Two-proportion z-test (Fisher exact when counts are small) |
| Rate change, paired items | McNemar |
| Continuous feature change | Two-sample Kolmogorov–Smirnov (or Mann–Whitney for location shift) |
| Category distribution change | Chi-square homogeneity; JSD as effect size |
| Multivariate distribution change | MMD with permutation test; classifier two-sample test |
| Many tests at once | Benjamini–Hochberg FDR |
| Slow drift in a stream | EWMA / CUSUM |
| Agent reliability | pass^k = C(c,k)/C(n,k) |
| Uncertainty on any statistic | Bootstrap over samples (≥1,000 resamples) |

### 6.4 Meaning groups (local and deterministic)
- Use a pinned local NLI model (for example, a DeBERTa-MNLI checkpoint). Two outputs belong to the same group if each entails the other ([Kuhn et al., 2023](https://arxiv.org/abs/2302.09664); [Farquhar et al., 2024](https://www.nature.com/articles/s41586-024-07421-0)).
- Group greedily: compare each output only with one representative per existing group. This costs about n × (number of groups) comparisons instead of n².
- Validate the grouping on 50+ hand-labelled same/different pairs per task type, and record the agreement in the control record.
- NLI models handle short text well. For long outputs, group on the extracted final answer or on the claims, not on the whole text.

### 6.5 Sample size (minimum detectable change)
Samples per side for a two-proportion test at α = 0.05 (two-sided) and power = 0.80 (computed):

| Baseline rate | Changed rate | Samples per side |
|---|---|---|
| 0.80 | 0.60 | 81 |
| 0.90 | 0.80 | 199 |
| 0.99 | 0.95 | 284 |
| 0.50 | 0.40 | 387 |
| 0.95 | 0.90 | 434 |

Formula: n = [z₁₋α/₂·√(2p̄(1−p̄)) + z₁₋β·√(p₁(1−p₁)+p₂(1−p₂))]² / (p₁−p₂)².

Choose the smallest change you need to catch, then size probes × samples to meet it. Pooled counts assume the change affects the pooled probes in the same direction and that samples are independent. Check both assumptions with the fault injections in §7.

### 6.6 Cost
- Cost per canary run = Σ over probes of k × (input tokens × input price + output tokens × output price).
- Monthly cost = runs per month × cost per canary run, plus D6 agent trials at their own token cost.
- Local NLI and embedding compute is CPU-scale and is not usually the binding cost.
- If the cost is too high, cut representative probes before tripwires (tripwires carry the most signal per token), and cut k only as far as §6.5 allows.
- D4 has no inference cost. If budget forces a choice, D4 + D1 + D2 is the minimum viable set.

### 6.7 Frequency
- Cadence = the longest undetected-change window the business can accept for that system.
- Add unscheduled runs on every deploy (P2) and every provider notice (D8).

---

## 7. Testing the controls (design effectiveness)

A control that has never caught a known fault is unproven. Run this suite at go-live and after any change to the controls themselves.

| Injected fault | Simulates | Controls expected to fire |
|---|---|---|
| Add one hidden system line (e.g. "Answer concisely.") in the gateway | S3 | D2 (certain), D3 structural (length) |
| Add a hidden instruction to append a safety disclaimer | S3 / S4 | D2, D3 structural (phrase hits) |
| Swap to a sibling model snapshot | S1 | D1, D3 (all layers), D5 |
| Raise temperature | S5 / S2-like | D3 meaning groups, P3 metrics on tripwires |
| Truncate outputs at 70% length | S4 | P4, D3 structural (`stop_reason`, length) |
| Alter one recorded tool output | S6 | D6 live versus replay divergence |
| Replace the embedding model file | S7 | D7 |

For each fault, record which controls detected it, how many probes and samples it took, and the detection delay. The smallest injected change that is detected reliably is that control's **demonstrated** minimum detectable change. Report that figure to stakeholders, not the theoretical figure from §6.5.

**Operating effectiveness:** retain run logs, alert tickets and C1/C3 records as evidence that the controls ran and that alerts were handled.

---

## 8. Changes to existing files in this folder

| File / item | Action | Reason |
|---|---|---|
| `semantic_similarity_guide.md`: "at temperature=0, most models are deterministic" | **Remove** | Not reliable: inference is non-deterministic at temperature 0 because of batching ([Thinking Machines, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). Hashes are audit evidence, not reproducibility proof |
| `semantic_similarity_guide.md`: "the hash log is the only way to detect [a model update]" | **Replace** with D1 + D2 + D3 | Hashes differ call to call anyway, so they cannot detect drift |
| `dev_check.py` ROUGE-L thresholds (0.05 / 0.15 / 0.30) and `prod_monitor.py` BERTScore thresholds (0.01 / 0.03), "3 consecutive runs", "15–30 examples" | **Replace** with §6.2 calibration and §6.5 sizing | These were starting guesses with no measurement behind them |
| BERTScore of one output against one reference | **Demote** to one feature in the D3 structural layer | Single-sample comparisons are dominated by sampling noise. Drift is a property of distributions |
| `semantic_similarity.py` cosine interpretation bands | **Recalibrate** with §6.2 or remove | Same issue as the other fixed thresholds |
| `log_inference` / `verify_inference` (hash logging), version pinning | **Keep**, extend into D1 | Correct for audit |
| ROUGE-L smoke test | **Keep** as a fast developer check only, not as a control | Fast and cheap; not a gate |

`rules.md` #2 ("treat prompt and model changes as deployments") is implemented by P1 + P2. #3 ("log the assembled context") is implemented by D1.

---

## 9. What these controls cannot catch

- **Changes that only affect inputs your probes don't cover.** D4 reduces this risk but cannot attribute cause.
- **Hidden layers already present when the baseline was taken.** The controls detect change, not original state.
- **Changes smaller than the demonstrated minimum detectable change** (§7). These are invisible by construction. State the figure explicitly.
- **Changes targeted at, or conditional on, specific content.** Example: a filter that only acts on one topic and never on your probes.
- **Consistent but wrong behaviour.** Stable outputs can be stably wrong. Only D5, and ultimately human review, address correctness.

---

## References
- Farquhar et al., *Detecting hallucinations in large language models using semantic entropy*, Nature 2024 — https://www.nature.com/articles/s41586-024-07421-0
- Kuhn, Gal, Farquhar, *Semantic Uncertainty*, ICLR 2023 — https://arxiv.org/abs/2302.09664
- Gretton et al., *A Kernel Two-Sample Test* (MMD), JMLR 2012 — https://jmlr.org/papers/v13/gretton12a.html
- Lopez-Paz & Oquab, *Revisiting Classifier Two-Sample Tests*, ICLR 2017 — https://arxiv.org/abs/1610.06545
- Yao et al., *τ-bench* (pass^k), 2024 — https://arxiv.org/abs/2406.12045
- Thinking Machines, *Defeating Nondeterminism in LLM Inference*, 2025 — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- Anthropic, *A postmortem of three recent issues*, 2025 — https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues
- Chen, Zaharia, Zou, *How is ChatGPT's behavior changing over time?*, 2023 — https://arxiv.org/abs/2307.09009
- NIST/SEMATECH e-Handbook of Statistical Methods, ch. 6 (process control) — https://www.itl.nist.gov/div898/handbook/
