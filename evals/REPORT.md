# Evaluation report

30 cases x 3 independent runs = 90 trajectories, agent `gpt-5.4-nano`, judge `gpt-5.6-luna` x1, cassettes `off`.

Every per-run trace is committed under `evals/results/traces/`, so each number below can be traced to the run that produced it. Regenerate with `make report`; CI asserts the result is byte-identical to what is committed.

## Headline

| Metric | Value | 95% CI | What it means |
|---|---|---|---|
| **pass@1** | **96.7%** | [83–99%] | a single attempt succeeds |
| **pass^3** | **93.3%** | [79–98%] | *all 3* attempts succeed — the reliability number |
| (pass@1)^3 | 90.3% | — | what pass^3 would be if every case behaved identically |
| independence gap | +0.030 | — | positive means outcomes are deterministic per case, not random per run |

## By tier

| Tier | Cases | pass@1 | 95% CI | Mean steps | Mean cost |
|---|---|---|---|---|---|
| easy | 12 | 100.0% | [76–100%] | 2.0 | $0.00174 |
| multi_hop | 11 | 93.9% | [62–98%] | 2.0 | $0.00233 |
| adversarial | 7 | 95.2% | [65–100%] | 3.4 | $0.00364 |

## Tool-call accuracy

| Metric | Value | What it uniquely catches |
|---|---|---|
| Tool precision | 36.8% | redundant or off-plan calls |
| Tool recall | 100.0% | skipped hops, or answering from memory without looking anything up |
| Tool F1 | 47.6% | nothing on its own — it hides P/R asymmetry, which is why both are above |
| Order score (LCS) | 100.0% | doing hop 2 before hop 1, which implies guessing |
| Step efficiency | 67.2% | serialisation waste: same F1, 3x the latency |
| Forbidden-tool rate | 0.0% | relevance-detection failures, e.g. searching the web for arithmetic |
| Citation resolution | 97.7% | **fabricated sources** — invisible to every metric above and to a careless judge |
| Invented-URL rate | — | URLs in the answer that were never retrieved |

**Read tool precision with care.** `expected_tools` names at most one call for 81 of these 90 runs (mean 0.93 expected against 4.37 actual), while the agent is designed to fan out several parallel searches in a single round. Precision is matched/n_actual, so a correct four-way fan-out against a one-call expectation scores 25% without a single wasted call. The figure above is therefore a lower bound that mixes genuine redundancy with the dataset's own thinness, and it is not evidence of waste on its own — **Where the rounds go** below measures that directly instead.

The order score is computed on the 3 run(s) that declare a `tool_order`, so it is a check that nothing is out of sequence on those cases rather than a rate with any sampling behind it.

## Graph coverage

Counted from the 90 `full` traces. Anything marked never exercised is something no number in this report speaks for.

| Node | Runs that reached it |
|---|---|
| `plan` | 90/90 |
| `act` | 90/90 |
| `observe` | 90/90 |
| `compact` | 0/90 **— never exercised** |
| `reflect` | 87/90 |
| `finalize` | 90/90 |

| Route taken | Times |
|---|---|
| `observe` → `reflect` | 194 |
| `observe` → `compact` | 0 **— never exercised** |
| `observe` → `finalize` | 16 |
| `reflect` → `act` | 119 |
| `reflect` → `plan` | 1 |
| `reflect` → `finalize` | 74 |

| Tool | Calls |
|---|---|
| `web_search` | 328 |
| `fetch_page` | 49 |
| `calculator` | 16 |
| `code_execution` | 0 **— never exercised** |
| `file_ops` | 0 **— never exercised** |

Never exercised by this evaluation: `compact`, `code_execution`, `file_ops`. The separate coverage suite runs each of them deliberately, and is reported apart from these quality numbers so it cannot move them.

### The coverage suite

_The coverage suite has not been run yet._

## Where the rounds go

| | Count | Share |
|---|---|---|
| Agent proposed to stop; `reflect` agreed (the run's last step) | 46 | 21.9% |
| Agent proposed to stop; `reflect` overruled it | 27 | 12.9% |
| Ran tools and found nothing new | 34 | 16.2% |
| Ran tools and found new sources | 103 | 49.0% |
| Searches repeating an earlier query verbatim | 0 | 0.0% |
| Tool calls returning no new source | 30 | 7.6% |

Across 210 rounds the agent made 393 tool calls and repeated a query 0 times. It is not looping on near-identical searches.

46 rounds are the agent saying it can answer and `reflect` agreeing — the graph needs one more `act` call after the last search just to say so. That is a structural cost of the topology, not a misjudgement.

27 rounds are the pathology: the agent proposed to stop and `reflect` sent it back with nothing to do. By tier: adversarial 27. Where they occur is the diagnosis — on questions that cannot be answered, `reflect` can always name a missing piece, because the missing piece does not exist.

## Cost and latency

| | Mean | p50 | p95 |
|---|---|---|---|
| Cost per run | $0.00240 | $0.00207 | $0.00423 |
| Latency | 13.4s | 11.4s | 25.7s |
| Steps | 2.3 | 2 | 8 |

Total for this run: agent $0.2161 + judge $0.1104 = **$0.3265** over 90 runs in 1412.7s.

## Budget behaviour

16 of 90 runs stopped early. Every one still returned an answer — the graph has no path to END that skips synthesis, and `finalize` falls back to an LLM-free assembly from state when the budget is genuinely gone.

| Reason | Runs |
|---|---|
| max_steps | 5 |
| tool_cap | 11 |

## Where it fails

Of 3 failed runs:

| Failure mode | Count |
|---|---|
| wrong_value | 3 |

**2 of 30 cases account for every failed run.** 0 fail on all 3 runs — a capability gap rather than variance, and the place to start. The rest are flaky and would partly average away with more runs.

| Case | Tier | Failed | Pattern | Judge failure modes | Early exits |
|---|---|---|---|---|---|
| `multi-011` | multi_hop | 2/3 | most runs | wrong_value | — |
| `adv-003` | adversarial | 1/3 | one run | wrong_value | tool_cap |

1 of the failing cases hit a per-tool call cap rather than deciding they were finished (`adv-003`). That is the over-searching this report measures elsewhere, showing up as a failure rather than merely as wasted spend.

## Is the judge any good?

_All 90 hand labels describe answers that have since been re-run, so none of them grades the text committed here. Judge-vs-human agreement is **not reported** rather than computed against answers no human read — a stale comparison still returns a number, which is what makes it dangerous. Re-label a sample of the current trajectories with `python -m evals.label` to restore it._

_Until then, every judge-derived figure in this report is unvalidated._

## Against a baseline

### Against a single round of search

> **Not evidence: these variants searched with different engines.** The full agent sent 0% of its searches somewhere other than Tavily; `baseline` sent 28%. Any difference below mixes the agent with the search engine, so no paired statistic is computed. Re-run both with the same `--search-backend`.

| Variant | pass@1 | searches off Tavily |
|---|---|---|
| baseline (one round of parallel search, then answer) | 93.3% | 28% |
| full agent | 96.7% | 0% |

### Does letting reflect overrule a stop earn its cost?

> **Not evidence: these variants searched with different engines.** The full agent sent 0% of its searches somewhere other than Tavily; `no_overrule` sent 100%. Any difference below mixes the agent with the search engine, so no paired statistic is computed. Re-run both with the same `--search-backend`.

| Variant | pass@1 | searches off Tavily |
|---|---|---|
| no_overrule (act's decision to stop is final) | 90.0% | 100% |
| full agent | 96.7% | 0% |


## Reading these numbers honestly

- **n=30 is small.** Every confidence interval above is wide, and any difference smaller than about 15 points is not distinguishable from noise.
- **pass^3 from 3 runs is a one-bit measurement per case** — an all-3-succeed rate, not a precise reliability estimate.
- **Self-preference is not controlled.** Judge and agent are both OpenAI models; they are different generations, not different families. A judge from another provider would be needed to rule it out.
- **Re-running produces different numbers.** The agent is non-deterministic and the web changes underneath it. The committed traces are the evidence, not a promise of reproducibility.

_Generated from `evals/results/` by `python -m evals.report`._
