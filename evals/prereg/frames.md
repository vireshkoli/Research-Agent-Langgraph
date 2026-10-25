# Pre-registration: full agent against baseline on FRAMES

Committed before any FRAMES run. Its purpose is to make the result impossible to
tune after the fact: the hypothesis, the sample, the test and the decision rule are
fixed here, and the git history shows they were fixed first.

## Why this exists

The 30 hand-written cases cannot say whether one variant beats another. Every
variant scores 100% on their easy tier, and a second pass of the comparison was
confounded when a search quota ran out mid-sweep. This evaluation is sized to
answer the question either way, and run so that nothing but the agent differs
between the two arms.

## Hypothesis

**H1:** the full agent's pass@1 on FRAMES differs from the baseline's.
Two-sided: the earlier passes pointed in opposite directions, so no direction is
assumed. The direction observed is reported, whichever it is.

## Arms

| Arm | What it is |
|---|---|
| `full` | plan → act → observe → reflect loop, as committed at the registration commit |
| `baseline` | plan once, one parallel round of search, answer |

`no_overrule` is not run: the question this answers is whether the loop earns its
cost, and a third arm would split a fixed budget across a question nobody asked.

## Sample

- **200 questions** from FRAMES (`google/frames-benchmark`, Apache-2.0), drawn by
  `evals/benchmarks/frames.py` and frozen in `evals/benchmarks/frames_sample.json`:
  stratified by primary reasoning type with largest-remainder allocation, seed
  20261025 — Multiple constraints 79, Numerical 71, Tabular 34, Temporal 12, Post
  processing 4.
- The source file's SHA-256 is recorded with the sample; a changed upstream file
  is an error, not a different benchmark.
- **One run per question per arm.** For a difference between arms, more questions
  are worth more than repeat runs of the same ones.

## Power

For a paired binary outcome, McNemar's test needs, at 80% power and α = 0.05
two-sided:

| Discordance | 5-pt gap | 10-pt gap | 15-pt gap |
|---|---|---|---|
| 15% | 469 | 116 | 50 |
| 25% | 783 | **194** | 85 |
| 35% | 1097 | 273 | 120 |

Discordance — the share of questions the arms disagree on — is expected near 25%
on multi-hop questions answered around 50–70% correctly. **n = 200 detects a gap of
about 10 points.** A smaller gap may not reach significance, and in that case the
result is still informative: the 95% interval on the gap at n = 200 is about
±5–8 points depending on discordance, which bounds any difference that exists.

## Primary outcome and test

- **Outcome:** pass@1 — the judge says CORRECT and every deterministic gate passes,
  exactly as for the hand-written set.
- **Test:** exact two-sided McNemar on the paired outcomes.
- **Interval:** seeded percentile bootstrap over questions for the difference in
  pass@1 (2,000 resamples, seed 42).
- **α = 0.05.** One primary test; no correction needed for it. Everything else
  below is secondary and labelled as such.

## Decision rule

- **p < 0.05:** report a significant difference, in the direction observed, with
  its interval.
- **p ≥ 0.05:** report that no difference was detected, and that the interval
  bounds any real difference to within its limits. Not "the arms are equal".
- **No re-running to move p.** The result of the run below is the result.

## Conditions held fixed across arms

- **Search engine:** DuckDuckGo, pinned with `--search-backend ddgs`; a pinned search
  fails rather than switching engines. Tavily's monthly quota is exhausted, and at
  ~1,800 searches this run would exceed it in any case. The live demo uses Tavily;
  this measures the agent, with the engine held constant.
- **Answer-key exclusion:** results and fetches from huggingface.co, hf.co,
  github.com, githubusercontent.com, kaggle.com and paperswithcode.com are dropped,
  because FRAMES' own answers are published there. Every trace is scanned
  afterwards and the count of hits reported; the target is zero.
- **Prompts:** the `prompt_hash` at the registration commit, recorded in every trace.
- **Models:** agent `gpt-5.4-nano`, judge `gpt-5.6-luna`, same rubric as the home set
  plus a per-case note that equivalent wording of the gold answer is correct.
- **Order:** both arms on the same day, back to back.

## Exclusions, decided now

A question is dropped from **both** arms only if either arm's run ended in
`internal_error` — a transport failure, not agent behaviour. Every exclusion is
listed in the results with its reason. Nothing is dropped for being hard, wrong, or
inconvenient.

## Budget and the pilot

The first **10 questions in committed id order** run first under this exact
protocol, to measure real cost per question. They count toward the 200. The only
thing decided from the pilot is the sample size: if the measured cost means 200
questions would exceed the remaining budget, *n* is cut to the largest multiple of
10 that fits, taken in committed id order, and that cut is recorded here in a
follow-up commit **before** any run beyond the pilot. Pilot accuracy is not used for
anything.

## Secondary analyses (reported, not tested for significance)

- `gold_source_recall`: the share of each question's gold Wikipedia articles the run
  retrieved — separates "never found the page" from "found it and reasoned wrong".
- Cost and latency per arm.
- A second judge from another model family (`evals/judge_cross.py`) on the same
  answers, and both judges against a seeded random sample of owner-labelled answers.
- Per-reasoning-type pass rates, which at these strata sizes are descriptive only.
