# Evaluation report

30 cases x 3 independent runs = 90 trajectories, agent `gpt-5.4-nano`, judge `gpt-5.6-luna` x1, cassettes `off`.

Every per-run trace is committed under `evals/results/traces/`, so each number below can be traced to the run that produced it. Regenerate with `make report`; CI asserts the result is byte-identical to what is committed.

## Headline

| Metric | Value | 95% CI | What it means |
|---|---|---|---|
| **pass@1** | **87.8%** | [70–95%] | a single attempt succeeds |
| **pass^3** | **76.7%** | [59–88%] | *all 3* attempts succeed — the reliability number |
| (pass@1)^3 | 67.6% | — | what pass^3 would be if every case behaved identically |
| independence gap | +0.090 | — | positive means outcomes are deterministic per case, not random per run |

## By tier

| Tier | Cases | pass@1 | 95% CI | Mean steps | Mean cost |
|---|---|---|---|---|---|
| easy | 12 | 100.0% | [76–100%] | 2.4 | $0.00161 |
| multi_hop | 11 | 78.8% | [52–95%] | 2.4 | $0.00266 |
| adversarial | 7 | 81.0% | [49–97%] | 3.1 | $0.00294 |

## Tool-call accuracy

| Metric | Value | What it uniquely catches |
|---|---|---|
| Tool precision | 40.4% | redundant or off-plan calls |
| Tool recall | 100.0% | skipped hops, or answering from memory without looking anything up |
| Tool F1 | 50.2% | nothing on its own — it hides P/R asymmetry, which is why both are above |
| Order score (LCS) | 100.0% | doing hop 2 before hop 1, which implies guessing |
| Step efficiency | 64.9% | serialisation waste: same F1, 3x the latency |
| Forbidden-tool rate | 0.0% | relevance-detection failures, e.g. searching the web for arithmetic |
| Citation resolution | 97.7% | **fabricated sources** — invisible to every metric above and to a careless judge |
| Invented-URL rate | 0.0% | URLs in the answer that were never retrieved |

## Cost and latency

| | Mean | p50 | p95 |
|---|---|---|---|
| Cost per run | $0.00230 | $0.00195 | $0.00590 |
| Latency | 23.7s | 15.4s | 79.7s |
| Steps | 2.5 | 2 | 8 |

Total for this run: agent $0.2073 + judge $0.0596 = **$0.2670** over 90 runs in 2395.7s.

## Budget behaviour

19 of 90 runs stopped early. Every one still returned an answer — the graph has no path to END that skips synthesis, and `finalize` falls back to an LLM-free assembly from state when the budget is genuinely gone.

| Reason | Runs |
|---|---|
| internal_error | 1 |
| max_steps | 7 |
| tool_cap | 11 |

## Where it fails

Of 11 failed runs:

| Failure mode | Count |
|---|---|
| wrong_value | 8 |
| other | 3 |

## Is the judge any good?

Hand-labelled **90** judgements before looking at any judge output.

| Statistic | Value |
|---|---|
| Raw agreement | 96.7% |
| **Cohen's κ** | **0.824** (almost perfect) |
| κ 95% CI (bootstrap, seeded) | [0.580, 1.000] |
| Gwet's AC1 | 0.959 |
| Deflation, raw − κ | 14.3 pts |
| Coverage | 100.0% (0 excluded) |

| | Human CORRECT | Human INCORRECT |
|---|---|---|
| **Judge CORRECT** | 79 | 0 |
| **Judge INCORRECT** | 3 | 8 |

Raw agreement overstates κ by 14 points here. That gap is why raw agreement alone is not reported: it counts the agreement you would get by chance, which on skewed data is most of it. Gwet's AC1 is shown alongside because κ is unstable when one label dominates the marginals. (The literature this design follows measures 33-41 points of deflation on MT-Bench; the gap here is smaller because raw agreement is unusually high, not because the correction stopped mattering.)

**The judge never inflates a score.** It said CORRECT where the human said INCORRECT **0 times**; all 3 disagreements run the other way, with the judge stricter than the human. That is the direction an evaluation wants to fail in: every pass rate in this report is a floor, not a flattering estimate. A judge that erred the other way would quietly inflate the agent and nothing else here would catch it.

**Verbosity probe:** correlation between verdict and answer length is **r = -0.380** — the judge scored shorter answers higher. Folk wisdom says LLM judges reward length, and the 2026 study behind this design measured that bias below 0.011, so a value this size is worth naming rather than filing away. It is also confounded: this agent writes longest when it is hedging on a question it could not settle, so length here tracks difficulty as much as style. Treat it as a flag for a larger sample, not as evidence the judge prefers brevity.

## Against a baseline

### Against a single round of search

Compared on the **90 runs both variants completed** (adversarial, easy, multi_hop).

| Variant | pass@1 | Mean steps | Mean cost |
|---|---|---|---|
| baseline (one round of parallel search, then answer) | 94.4% | 1.0 | $0.00077 |
| **full agent** | **87.8%** | 2.5 | $0.00230 |

### Does letting reflect overrule a stop earn its cost?

> 20 runs of this variant were discarded: The OpenAI key was rotated while this variant was running. These runs failed authentication instantly (0 steps, $0.00, sub-second) and are transport failures, not agent behaviour. They are removed rather than scored. Affected cases: adv-001, adv-002, adv-003, adv-004, adv-005, adv-006, adv-007. The comparison below is therefore restricted to the runs that survived, and does not cover the adversarial tier.

Compared on the **70 runs both variants completed** (adversarial, easy, multi_hop).

| Variant | pass@1 | Mean steps | Mean cost |
|---|---|---|---|
| no_overrule (act's decision to stop is final) | 93.1% | 2.1 | $0.00199 |
| **full agent** | **90.3%** | 2.4 | $0.00213 |


## Reading these numbers honestly

- **n=30 is small.** Every confidence interval above is wide, and any difference smaller than about 15 points is not distinguishable from noise.
- **pass^3 from 3 runs is a one-bit measurement per case** — an all-3-succeed rate, not a precise reliability estimate.
- **Self-preference is not controlled.** Judge and agent are both OpenAI models; they are different generations, not different families. A judge from another provider would be needed to rule it out.
- **Re-running produces different numbers.** The agent is non-deterministic and the web changes underneath it. The committed traces are the evidence, not a promise of reproducibility.

_Generated from `evals/results/` by `python -m evals.report`._
