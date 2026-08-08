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

_No human labels committed yet, so judge-vs-human agreement is not reported. Until it is, treat every judge-derived number above as unvalidated._

## Against a baseline

_Ablations not run yet._

## Reading these numbers honestly

- **n=30 is small.** Every confidence interval above is wide, and any difference smaller than about 15 points is not distinguishable from noise.
- **pass^3 from 3 runs is a one-bit measurement per case** — an all-3-succeed rate, not a precise reliability estimate.
- **Self-preference is not controlled.** Judge and agent are both OpenAI models; they are different generations, not different families. A judge from another provider would be needed to rule it out.
- **Re-running produces different numbers.** The agent is non-deterministic and the web changes underneath it. The committed traces are the evidence, not a promise of reproducibility.

_Generated from `evals/results/` by `python -m evals.report`._
