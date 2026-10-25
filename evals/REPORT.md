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

**Read tool precision with care.** `expected_tools` names at most one call for 81 of these 90 runs (mean 0.93 expected against 4.03 actual), while the agent is designed to fan out several parallel searches in a single round. Precision is matched/n_actual, so a correct four-way fan-out against a one-call expectation scores 25% without a single wasted call. The figure above is therefore a lower bound that mixes genuine redundancy with the dataset's own thinness, and it is not evidence of waste on its own — **Where the rounds go** below measures that directly instead.

The order score is computed on the 3 run(s) that declare a `tool_order`, so it is a check that nothing is out of sequence on those cases rather than a rate with any sampling behind it.

## Where the rounds go

| | Count | Share |
|---|---|---|
| Rounds where `act` emitted no tool calls | 77 | 33.8% |
| Rounds that ran tools and found nothing new | 45 | 19.7% |
| Productive rounds | 106 | 46.5% |
| Searches repeating an earlier query verbatim | 2 | 0.7% |
| Tool calls returning no new source | 22 | 6.1% |

Across 228 rounds the agent made 363 tool calls, of which 2 repeated a query it had already issued. **It is not looping on near-identical searches, and it is not making large numbers of useless calls.** What it does do is spend 77 rounds (34% of all of them) calling no tools at all. `act` emits no calls when it believes it can answer, so those rounds are the agent asking to stop and being sent back by `reflect`, paying for one `act` call and one `reflect` call each time to do nothing.

That is a different defect from over-searching and it has a different fix: not fewer searches, but honouring a stop that has already been proposed.

## Cost and latency

| | Mean | p50 | p95 |
|---|---|---|---|
| Cost per run | $0.00230 | $0.00195 | $0.00590 |
| Latency | 23.7s | 15.4s | 79.7s |
| Steps | 2.5 | 2 | 8 |

Total for this run: agent $0.2073 + judge $0.1193 = **$0.3266** over 90 runs in 2395.7s.

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

**7 of 30 cases account for every failed run.** 1 fails on all 3 runs — a capability gap rather than variance, and the place to start. The rest are flaky and would partly average away with more runs.

| Case | Tier | Failed | Pattern | Judge failure modes | Early exits |
|---|---|---|---|---|---|
| `multi-011` | multi_hop | 3/3 | every run | other, wrong_value | — |
| `adv-007` | adversarial | 2/3 | most runs | wrong_value | tool_cap |
| `multi-002` | multi_hop | 2/3 | most runs | wrong_value | — |
| `adv-003` | adversarial | 1/3 | one run | other | tool_cap |
| `adv-006` | adversarial | 1/3 | one run | other | internal_error |
| `multi-006` | multi_hop | 1/3 | one run | wrong_value | tool_cap |
| `multi-007` | multi_hop | 1/3 | one run | wrong_value | — |

1 of those failed runs exited with `internal_error` — a transport failure rather than the agent being wrong. It is left in the table rather than quietly dropped, but not treated as evidence about capability.

3 of the failing cases hit a per-tool call cap rather than deciding they were finished (`adv-003`, `adv-007`, `multi-006`). That is the over-searching this report measures elsewhere, showing up as a failure rather than merely as wasted spend.

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

**Paired comparison.** The difference in pass@1 is +6.7 points in favour of `baseline`, with a 95% bootstrap interval over cases of [+0.0, +14.4] points. 3.8% of resamples failed to favour it. McNemar's exact test on the 12 discordant (case, run) pairs — 3 where only the full agent succeeded, 9 where only `baseline` did — gives p = 0.146.

**Unique wins.** The full agent reliably solves **0** cases that `baseline` does not; `baseline` reliably solves **4** that the full agent does not. A variant that costs more and wins nothing outright is not trading accuracy for cost — it is paying for nothing.

### Does letting reflect overrule a stop earn its cost?

> 21 runs of this variant were re-run rather than left missing (adv-001, adv-002, adv-003, adv-004, adv-005, adv-006, adv-007). The adversarial tier was lost when the OpenAI key was rotated mid-run on 2026-08-09 and was re-run in full on 2026-10-25 against the same dataset, prompts and models. The surviving easy and multi-hop runs are from 2026-08-08/09. The tiers are therefore not contemporaneous: the web moved between the two dates, and adv-001 run 1 — the single adversarial run that survived the rotation — was re-run and replaced rather than mixed with the new ones, so every adversarial row here matches the trace committed beside it.

Compared on the **90 runs both variants completed** (adversarial, easy, multi_hop).

| Variant | pass@1 | Mean steps | Mean cost |
|---|---|---|---|
| no_overrule (act's decision to stop is final) | 91.1% | 2.2 | $0.00219 |
| **full agent** | **87.8%** | 2.5 | $0.00230 |

**Paired comparison.** The difference in pass@1 is +3.3 points in favour of `no_overrule`, with a 95% bootstrap interval over cases of [-4.4, +11.1] points. 22.9% of resamples failed to favour it. McNemar's exact test on the 11 discordant (case, run) pairs — 4 where only the full agent succeeded, 7 where only `no_overrule` did — gives p = 0.549.

**Unique wins.** The full agent reliably solves **1** case that `no_overrule` does not; `no_overrule` reliably solves **3** that the full agent does not. The richer variant does win ground of its own here, so the extra cost buys something — the question is whether it buys enough.


## Reading these numbers honestly

- **n=30 is small.** Every confidence interval above is wide, and any difference smaller than about 15 points is not distinguishable from noise.
- **pass^3 from 3 runs is a one-bit measurement per case** — an all-3-succeed rate, not a precise reliability estimate.
- **Self-preference is not controlled.** Judge and agent are both OpenAI models; they are different generations, not different families. A judge from another provider would be needed to rule it out.
- **Re-running produces different numbers.** The agent is non-deterministic and the web changes underneath it. The committed traces are the evidence, not a promise of reproducibility.

_Generated from `evals/results/` by `python -m evals.report`._
