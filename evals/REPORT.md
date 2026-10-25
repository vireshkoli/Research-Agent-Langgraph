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

## Where the rounds go

| | Count | Share |
|---|---|---|
| Rounds where `act` emitted no tool calls | 73 | 34.8% |
| Rounds that ran tools and found nothing new | 34 | 16.2% |
| Productive rounds | 103 | 49.0% |
| Searches repeating an earlier query verbatim | 0 | 0.0% |
| Tool calls returning no new source | 30 | 7.6% |

Across 210 rounds the agent made 393 tool calls, of which 0 repeated a query it had already issued. **It is not looping on near-identical searches, and it is not making large numbers of useless calls.** What it does do is spend 73 rounds (35% of all of them) calling no tools at all. `act` emits no calls when it believes it can answer, so those rounds are the agent asking to stop and being sent back by `reflect`, paying for one `act` call and one `reflect` call each time to do nothing.

That is a different defect from over-searching and it has a different fix: not fewer searches, but honouring a stop that has already been proposed.

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

Hand-labelled **90** judgements before looking at any judge output.

| Statistic | Value |
|---|---|
| Raw agreement | 92.2% |
| **Cohen's κ** | **0.331** (fair) |
| κ 95% CI (bootstrap, seeded) | [-0.033, 0.676] |
| Gwet's AC1 | 0.912 |
| Deflation, raw − κ | 59.1 pts |
| Coverage | 100.0% (0 excluded) |

| | Human CORRECT | Human INCORRECT |
|---|---|---|
| **Judge CORRECT** | 81 | 6 |
| **Judge INCORRECT** | 1 | 2 |

Raw agreement overstates κ by 59 points here. That gap is why raw agreement alone is not reported: it counts the agreement you would get by chance, which on skewed data is most of it. Gwet's AC1 is shown alongside because κ is unstable when one label dominates the marginals. (The literature this design follows measures 33-41 points of deflation on MT-Bench; the gap here is smaller because raw agreement is unusually high, not because the correction stopped mattering.)

**The judge passed 6 run(s) the human failed**, against 1 in the other direction. Pass rates here may therefore be slightly optimistic.

**Verbosity probe:** correlation between verdict and answer length is **r = -0.222** — the judge scored shorter answers higher. Folk wisdom says LLM judges reward length, and the 2026 study behind this design measured that bias below 0.011, so a value this size is worth naming rather than filing away. It is also confounded: this agent writes longest when it is hedging on a question it could not settle, so length here tracks difficulty as much as style. Treat it as a flag for a larger sample, not as evidence the judge prefers brevity.

## Against a baseline

### Against a single round of search

Compared on the **90 runs both variants completed** (adversarial, easy, multi_hop).

| Variant | pass@1 | Mean steps | Mean cost |
|---|---|---|---|
| baseline (one round of parallel search, then answer) | 93.3% | 1.0 | $0.00082 |
| **full agent** | **96.7%** | 2.3 | $0.00240 |

**Paired comparison.** The difference in pass@1 is -3.3 points in favour of `baseline`, with a 95% bootstrap interval over cases of [-7.8, +1.1] points. 94.9% of resamples failed to favour it. McNemar's exact test on the 5 discordant (case, run) pairs — 4 where only the full agent succeeded, 1 where only `baseline` did — gives p = 0.375.

**Unique wins.** The full agent reliably solves **3** cases that `baseline` does not; `baseline` reliably solves **1** that the full agent does not. The richer variant does win ground of its own here, so the extra cost buys something — the question is whether it buys enough.

### Does capping reflect's overrules help?

`gated` is identical to the full agent except that `reflect` may overrule a proposed stop at most `max_reflect_overrules` times, after which the stop is final. It was designed against the traces of an earlier run, where 34% of rounds were the agent proposing to stop and being sent back. It does not work, and this is the measurement that says so.

Compared on the **90 runs both variants completed** (adversarial, easy, multi_hop).

| Variant | pass@1 | Mean steps | Mean cost |
|---|---|---|---|
| gated (reflect may overrule a stop once) | 90.0% | 2.1 | $0.00229 |
| **full agent** | **96.7%** | 2.3 | $0.00240 |

**Paired comparison.** The difference in pass@1 is -6.7 points in favour of `gated`, with a 95% bootstrap interval over cases of [-13.3, -1.1] points. 100.0% of resamples failed to favour it. McNemar's exact test on the 6 discordant (case, run) pairs — 6 where only the full agent succeeded, 0 where only `gated` did — gives p = 0.031.

**Unique wins.** The full agent reliably solves **3** cases that `gated` does not; `gated` reliably solves **0** that the full agent does not. The richer variant does win ground of its own here, so the extra cost buys something — the question is whether it buys enough.

### Does letting reflect overrule a stop earn its cost?

Compared on the **90 runs both variants completed** (adversarial, easy, multi_hop).

| Variant | pass@1 | Mean steps | Mean cost |
|---|---|---|---|
| no_overrule (act's decision to stop is final) | 90.0% | 2.6 | $0.00233 |
| **full agent** | **96.7%** | 2.3 | $0.00240 |

**Paired comparison.** The difference in pass@1 is -6.7 points in favour of `no_overrule`, with a 95% bootstrap interval over cases of [-14.4, +0.0] points. 99.0% of resamples failed to favour it. McNemar's exact test on the 8 discordant (case, run) pairs — 7 where only the full agent succeeded, 1 where only `no_overrule` did — gives p = 0.070.

**Unique wins.** The full agent reliably solves **4** cases that `no_overrule` does not; `no_overrule` reliably solves **1** that the full agent does not. The richer variant does win ground of its own here, so the extra cost buys something — the question is whether it buys enough.


## Reading these numbers honestly

- **n=30 is small.** Every confidence interval above is wide, and any difference smaller than about 15 points is not distinguishable from noise.
- **pass^3 from 3 runs is a one-bit measurement per case** — an all-3-succeed rate, not a precise reliability estimate.
- **Self-preference is not controlled.** Judge and agent are both OpenAI models; they are different generations, not different families. A judge from another provider would be needed to rule it out.
- **Re-running produces different numbers.** The agent is non-deterministic and the web changes underneath it. The committed traces are the evidence, not a promise of reproducibility.

_Generated from `evals/results/` by `python -m evals.report`._
