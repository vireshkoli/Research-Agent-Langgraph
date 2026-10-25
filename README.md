# Tool-Using Research Agent

A ReAct agent on an explicit LangGraph state graph: it decomposes a question, calls tools in a loop,
and returns a **cited** answer — with hard budgets on steps, wall clock and dollars, so it can never
run away.

**[Live demo →](https://vireshk-research-agent.hf.space)** · Hugging Face Space, alongside [production-rag-platform](https://huggingface.co/spaces/vireshk/production-rag-platform) · first load after 48h idle takes ~40s

![The agent planning, searching, checking coverage and returning a cited answer](docs/demo.gif)

*Two sub-questions, four parallel searches, a coverage check, the calculator for the arithmetic,
then a cited answer — 405B − 175B = 230B. Real run, unedited apart from being sped up 1.8×.*

Budgets are the feature, not a safety net bolted on afterwards. When one trips, the run returns a
partial answer with whatever it did establish — never an error, never nothing.

---

## Measured results

30 cases × 3 independent runs = **90 trajectories**, agent `gpt-5.4-nano`, judge `gpt-5.6-luna`.
Every per-run trace is committed under [`evals/results/traces/`](evals/results/traces/), so every
number below can be traced back to the run that produced it.

| Metric | Value | 95% CI |
|---|---|---|
| **pass@1** — a single attempt succeeds | **96.7%** | [83–99%] |
| **pass^3** — *all three* attempts succeed | **93.3%** | [79–98%] |
| Tool recall | 100.0% | — |
| Tool precision | 36.8% | see the caveat below |
| Citation resolution (cited ids that exist) | 97.7% | — |
| Invented-URL rate | 0.0% | — |
| Cost per run | $0.0024 | p95 $0.0042 |
| Latency | 11.4s p50 | 25.7s p95 |

Full breakdown, per-tier results and failure modes: **[evals/REPORT.md](evals/REPORT.md)**.

### The headline finding: the loop's problem was its prompt, not its architecture

The point of an evaluation is to be able to be wrong, and this one has now been wrong about its own
agent — in a way worth keeping on the record, because the correction is the result.

**Pass one.** A `baseline` variant — plan once, issue every search in one parallel round, answer —
beat the full iterative agent on every tier: **94.4% against 87.8%**, three times cheaper. The
conclusion on the evidence then available was that iterative refinement was not merely failing to
pay for itself but actively destructive. Those results and their traces are tagged
[`v0.2.0-pre-gate`](../../releases/tag/v0.2.0-pre-gate).

**What the traces actually said.** The obvious reading is that the loop searches too much, and that
turned out to be false. Only **2 of 305 searches** repeated a query and only 6% of calls returned
nothing new. What the loop did was stop calling tools and keep going anyway: **77 of 228 rounds —
34% — ran no tools at all.** `act` emits no calls when it believes it can answer, so each of those
was the agent asking to stop, `reflect` overruling, and `act` proposing the same stop again. One
adversarial case did this eight times consecutively.

Two lines of prompt were responsible, and both are the kind of defect no test catches:

- `SYSTEM` said *"prefer several specific searches over one broad one"* and then, four lines later,
  *"do not keep searching to feel thorough."* The first is about the width of one round and now
  says so.
- `REFLECT_SYSTEM` listed its two failure modes *"in order of how common they are"* and put
  **stopping early first**. These very runs measured the opposite. The order is now inverted, the
  measurement is quoted in the prompt, and a proposed stop puts the burden on reflect to name the
  specific missing evidence.

**Pass two, after fixing those two lines and changing nothing else:**

| Tier | Full agent | Baseline | `no_overrule` |
|---|---|---|---|
| easy | 100.0% | 100.0% | 100.0% |
| multi-hop | **93.9%** | 81.8% | 81.8% |
| adversarial | 95.2% | **100.0%** | 85.7% |
| **overall pass@1** | **96.7%** | 93.3% | 90.0% |
| pass^3 | **93.3%** | 86.7% | 83.3% |
| mean steps | 2.33 | 1.00 | 2.56 |
| mean cost | $0.0024 | $0.0008 | $0.0023 |

The loop now beats the baseline it used to lose to. **That difference is not significant** —
+3.3 points, 95% paired interval [−1.1, +7.8], McNemar p = 0.375 — and the baseline remains three
times cheaper and still perfect on the adversarial tier, where it cannot be talked out of stopping.
What is defensible is narrower and more useful than the original claim: **the architecture was never
the problem, and the measured failure was a prompt that told the model to fear the wrong mistake.**

**`reflect` now earns its place, and the ablation is how we know.** `no_overrule` makes act's stop
decision final, removing exactly the authority that looked harmful in pass one. It scores **90.0%
against the full agent's 96.7%** — +6.7 points, paired interval [+0.0, +14.4], McNemar p = 0.070,
and the full agent reliably solves 4 cases it does not against 1 the other way. In pass one this
same ablation *beat* the full agent. Nothing about the node changed; its instructions did.

**One comparison here is confounded and should not be read as a clean before/after.** The 87.8% was
measured in August and the 96.7% today, so two and a half months of web drift sit inside it
alongside the prompt change. The three-way comparison in the table above was run back to back
against one prompt set and is sound; the across-time one is indicative only. Every trace records
the hash of the prompts that produced it, and `evals/REPORT.md` prints a banner if the committed
results ever stop matching the code.

**Is the judge trustworthy?** This finding rests on judge verdicts, so the judge was checked
against 90 hand labels made before any verdict was visible: **Cohen's κ = 0.824** (raw agreement
96.7%, Gwet's AC1 0.959). More usefully than the headline number, the errors are one-directional —
the judge said CORRECT where the human said INCORRECT **zero times**, and all three disagreements
are it being *stricter* than the human. Every pass rate here is therefore a floor rather than a
flattering estimate, which is the direction an evaluation should fail in.

**Caveats, stated rather than buried.** At 11 multi-hop and 7 adversarial cases the per-tier
intervals are very wide and no single tier difference here is significant on its own. Only one
comparison in this README is: capping reflect's overrules makes the agent worse. Everything else is
a direction, not a result, and this is 30 cases, not 300. The honest claim is *"on this benchmark,
with these prompts, the loop is no longer losing"* — not *"agent loops work"*, and emphatically not
the reverse claim this same benchmark supported two months ago.

`pass^3` is 93.3% against a pass@1 of 96.7%, and the gap between `pass^3` and `(pass@1)^3` is
**+0.030**. Because x^k is convex, that quantity is non-negative whenever per-case rates differ at
all, so a positive value means failures are **concentrated in a fixed set of cases** rather than
scattered randomly across runs. It has fallen from +0.090 as the overall rate rose, which is what
fixing a systematic defect looks like: the remaining failures are fewer and closer to genuine
run-to-run variance than to a fixed set of cases the agent cannot do.

---

## What it does

1. **plan** — decomposes the question into 1–5 searchable sub-questions.
2. **act** — the model picks tools. Tools that have hit their per-run cap are withdrawn from the
   schema rather than offered and refused.
3. **observe** — runs the calls in a thread pool, mints citable source ids, truncates observations,
   and counts consecutive failures for a circuit breaker.
4. **compact** — folds older steps into a rolling summary when the prompt grows past a threshold.
5. **reflect** — checks coverage against the sub-question list and decides continue / replan / finish.
6. **finalize** — synthesises a cited answer, or assembles one from state without an LLM if the
   budget is gone.

Tools: `web_search` (Tavily, with a circuit-broken `ddgs` fallback), `fetch_page`, `calculator`,
`code_execution` (sandboxed), `file_ops` (workspace-confined).

## Architecture

```mermaid
flowchart TD
    S([START])
    plan["plan · decompose into sub-questions"]
    act["act · LLM emits tool calls"]
    observe["observe · run tools · mint source ids"]
    compact["compact · summarise old steps"]
    reflect["reflect · coverage check"]
    finalize["finalize · cited synthesis"]
    E([END])
    r1{route_after_observe}
    r2{route_after_reflect}

    S --> plan
    plan --> act
    act --> observe
    observe --> r1
    r1 -->|budget verdict| finalize
    r1 -->|prompt over threshold| compact
    r1 -->|otherwise| reflect
    compact --> reflect
    reflect --> r2
    r2 -->|gaps remain| act
    r2 -->|plan is wrong| plan
    r2 -->|covered or budget| finalize
    finalize --> E
```

This diagram is generated by `research_agent.graph.mermaid()` from the same constants the graph is
wired from, so it cannot drift out of sync with the code. A test asserts every node appears in it.

### Design decisions

| Concern | Choice | Why |
|---|---|---|
| Graph construction | Hand-built `StateGraph`, not `create_react_agent` | The control flow *is* the artefact. A prebuilt agent hides both the routing and the eval surface. (`create_react_agent` is also deprecated as of LangGraph 1.0.) |
| Recording *why* a run stopped | One pure `budget_verdict(state, cfg)`, called twice | LangGraph's conditional edges are pure routers — they return a node name and cannot write state, so the edge that decides to bail cannot record the reason. The router calls it to route; `finalize` calls it again to write the reason. Being pure, they agree by construction. |
| Running out of money mid-run | A reserve withheld from the loop | A loop that spent to 100% of the cap would leave `finalize` unable to afford its own call and return nothing — the opposite of graceful degradation. |
| Losing citations to compaction | Sources live in state, not in message text | Compaction rewrites the scratchpad; the citation registry is a different key. No summariser mistake can reach it. A test proves this by feeding in a summary that names no sources at all. |
| Message format | Raw OpenAI dicts with `operator.add`, not `add_messages` | `llm.py` speaks the wire format directly, so `add_messages` would coerce dicts into objects that every call converts back. Its one real feature — ID-based dedup — is unused here. |
| Parallel tool calls | A thread pool *inside* `observe` | Parallel graph branches writing the same reducer-less key in one super-step raise `InvalidUpdateError`, and each branch burns a super-step against the recursion limit. |
| Hosting | A free ZeroGPU Space that never uses the GPU | Hugging Face gates cpu-basic Spaces behind PRO; ZeroGPU is the free route once an account is ~30 days old. ZeroGPU refuses to start without a `@spaces.GPU` function, so `app.py` declares one that is never called — the GPU is allocated on *call*, not on decoration, so nothing is reserved. Measured, not read from docs: the first attempt failed with exactly that error. |
| API surface | Responses, not Chat Completions | Measured, not assumed: Chat Completions returns a 400 for `tools` + `reasoning_effort` on both `gpt-5.4-nano` and `-mini`. Responses handles it. |

---

## Evaluation

Most agent repos ship a loop and a screenshot. The eval is the part this project is actually about,
so it is hand-rolled rather than delegated: `agentevals` has had no release in a year, Ragas changed
GitHub orgs and is deprecating its own `ToolCallAccuracy`, and OpenAI's hosted Evals platform shuts
down in November 2026. A few hundred owned lines outlive all three, and the metric definitions stay
legible instead of living behind someone else's `.evaluate()`.

**One rule governs the dataset**, enforced by the loader: every case carries at least one
deterministic anchor — a string that must (or must not) appear in a correct answer. If a correct
answer cannot be pinned that way, the case is unscoreable and the judge ends up measuring its own
taste. Cases that cannot meet that bar are cut.

**Tool-call accuracy** uses maximum bipartite matching between expected and actual calls, not greedy
left-to-right — greedy lets one actual call satisfy two expectations, inflating recall on exactly the
runs where the agent was lazy. Arguments are compared field by field with per-field semantics (seven
matchers: exact, contains_any/all, regex, numeric_close, lte/gte, any), not by serialising both sides
and comparing strings. All three degenerate cases are defined explicitly, because undefined edge
cases are where harnesses quietly return flattering numbers:

| Expected | Actual | P / R / F1 |
|---|---|---|
| 0 | 0 | 1.0 / 1.0 / 1.0 — correctly declined to use a tool |
| 0 | >0 | 0.0 / 1.0 / 0.0 |
| >0 | 0 | 1.0 / 0.0 / 0.0 |

**Ordering** is scored by longest common subsequence rather than Kendall tau, so an agent that got
the spine right and inserted an extra hop gets partial credit; Kendall penalises every inversion
equally and cannot tell those apart. **Forbidden-tool calls are reported separately from F1** — waste
and safety are not capability. **Citation resolution and invented-URL rate** are the cheapest
hallucination checks here and the only ones that catch a confident answer citing a source that does
not exist.

**The judge** is binary with an anchored rubric and four worked examples, and every committed
evaluation ran it at one sample per trajectory. It is validated against 90 hand labels instead of
against itself — agreement with a human is the evidence that matters, and self-consistency would
not have caught a judge that is confidently and consistently wrong. Unparseable judgements are
excluded and counted, never coerced to INCORRECT, since coercion biases the result in whichever
direction it goes.

Every rate carries a **95% Wilson interval**. At n=30 the normal approximation runs outside [0,1] and
is simply wrong, and a rate without an interval at this sample size is the first thing worth
questioning.

`make report` regenerates [evals/REPORT.md](evals/REPORT.md) from the committed results, and CI runs
it followed by `git diff --exit-code` — which proves mechanically that every number here comes from a
committed artefact rather than from something typed once by hand.

---

## Quickstart

```bash
git clone https://github.com/vireshkoli/Research-Agent-Langgraph.git
cd Research-Agent-Langgraph

uv sync --dev
cp .env.example .env          # add OPENAI_API_KEY; TAVILY_API_KEY is optional
uv run pre-commit install     # optional: ruff + gitleaks before each commit
make check                    # ruff + mypy + 272 tests, no API key needed

uv run python -m research_agent "Who won the 2024 Nobel Prize in Physics, and why?"
```

Watch a budget trip and still get an answer:

```bash
uv run python -m research_agent --max-steps 1 \
  "Which is larger: the largest Llama 3.1 model or GPT-3? Give both figures and the difference."
# → partial cited answer, early_exit_reason: max_steps, exit code 0
```

Other entry points:

```bash
make eval-dry                 # validates all 30 eval cases, spends $0
make report                   # regenerates evals/REPORT.md from committed results
docker compose up demo        # the same app in the local container, on localhost:7860
make deploy-hf                # push to the Hugging Face Space (needs HF_TOKEN)
uv run python -m research_agent.tools --demo   # exercise every tool once
```

---

## Known limitations

Written before anyone asks.

- **The loop's margin over the baseline is not significant.** It wins overall (96.7% vs 93.3%) and
  on multi-hop, but at +3.3 points with a paired interval of [−1.1, +7.8] and McNemar p = 0.375 that
  is not a result to lean on. The baseline is still 3× cheaper and still perfect on the adversarial
  tier. "The loop is better" is not supported; "the loop is no longer worse, and the prompt was the
  problem" is.
- **The headline reversed once, and could again.** This evaluation concluded the opposite in its
  first pass, and the thing that changed was two lines of prompt. An n=30 benchmark that flips on a
  prompt edit should be treated as a diagnostic instrument, not a leaderboard.
- **Tool precision understates the agent.** The 36.8% is matched/n_actual against
  `expected_tools`, which names at most one call for 81 of 90 runs while the agent is designed to
  fan out several parallel searches per round. A correct four-way fan-out against a one-call
  expectation scores 25% with nothing wasted. The number is a lower bound confounded with the
  dataset's own thinness, and `evals/REPORT.md` now measures redundancy directly instead of
  inferring it from precision. The order score is likewise computed on only the 3 runs that declare
  a `tool_order`.
- **The before/after across the prompt fix is confounded.** 87.8% was measured in August and 96.7%
  today, so web drift sits inside that gap alongside the prompt change. The four-way comparison
  between variants was run back to back against one prompt set and is sound; the across-time
  comparison is indicative only.
- **Judge costs were under-recorded and are now corrected.** Judge calls were priced at the 50%
  Batch rate while being issued synchronously, which halved every published judge figure. The
  committed results carry the corrected numbers, the discount is gone from the code, and the
  totals here reflect the real spend.
- **The judge's κ interval is wide.** Cohen's κ is 0.824 against 90 hand labels, but the bootstrap
  95% CI runs [0.580, 1.000] — "almost perfect" at the point estimate, only "moderate" at the lower
  bound. κ is unstable when one label dominates the marginals, which is why Gwet's AC1 (0.959) is
  reported beside it.
- **Self-preference is not controlled.** Judge and agent are both OpenAI models. They are different
  *generations*, not different *families*; ruling this out needs a judge from another provider.
- **n=30 is small.** Every interval above is wide. Differences under ~15 points are not
  distinguishable from noise.
- **Prompt caching does not engage.** Measured 6.6% cache hit rate. The stable prefix (system prompt
  plus question) is around 270 tokens, below OpenAI's 1024-token minimum for caching. An earlier cost
  model here assumed ~58% and was wrong; runs came in cheaper than budgeted anyway, for unrelated
  reasons.
- **`code_execution` is a resource guardrail, not a security boundary**, and its docstring says so at
  length. Real isolation (network namespaces, bubblewrap, gVisor) needs `CAP_SYS_ADMIN`, which
  unprivileged container hosts do not grant. What it does provide is a scrubbed environment — the
  child process never sees an API key — plus an import allowlist, self-imposed resource limits and a
  process-group kill. Do not point it at anything whose compromise would matter.
- **Two of my own eval cases were wrong**, and both were caught before publication. One scored a
  correct refusal as a failure because a substring anchor cannot distinguish quoting a prompt-injection
  payload from obeying it. The other asked how long before the Web went public *Snow Crash* was
  released, marking the agent wrong for answering 1993 (CERN's public-domain release) against a
  reference asserting 1991 — both defensible. That case was rewritten and re-run rather than having
  its rubric loosened after the fact; the substitution is recorded in `evals/results/full.json`.
- **The Space depends on a ZeroGPU quirk.** It runs free only because ZeroGPU accepts a declared-but-never-called `@spaces.GPU` function. If Hugging Face tightens that check the demo goes dark; the Dockerfile and `docker compose up demo` are the fallback, and the GIF above is the proof-of-life that survives either way.
- **Re-running produces different numbers.** The agent is non-deterministic and the web moves under
  it. The committed traces are the evidence, not a reproducibility promise.

## What I'd do next

1. **Find out why the middle is the worst place to be.** On the adversarial tier accuracy is not
   monotonic in rounds — `baseline` at 1.00 steps scores 100%, `no_overrule` at 2.48 steps scores
   85.7%, the full agent at 3.38 steps scores 95.2%. Either answer immediately or investigate
   properly; a truncated investigation returns a confident wrong value where a refusal was wanted,
   and `wrong_value` is indeed the dominant failure mode on exactly those cases. If that shape
   generalises it is a real result about when to let an agent stop, and it is the most interesting
   unexplained thing in this repo.
2. **Re-run the first pass against the fixed prompts on the same day.** That is the only way to
   separate the prompt fix from web drift, and it costs about $0.30.
3. **A larger labelled sample.** 90 items puts κ's lower bound at 0.580; a few hundred would
   tighten it enough to distinguish "almost perfect" from "moderate".
4. **A cross-family judge** (Claude or Gemini) to bound self-preference rather than disclaiming it.
5. **Grow the dataset to ~100 cases.** The interval width at n=30 is the binding constraint on saying
   anything is better than anything else.
6. **Make search results cheaper to read.** A rerank-and-truncate pass over search snippets before
   they enter the prompt would cut input tokens materially at these step counts.

---

## Repository layout

```
src/research_agent/
  llm.py         the single OpenAI wrapper: pricing, cost tracking, four spend guards
  graph.py       StateGraph wiring, both routers, the mermaid diagram
  budget.py      budget_verdict() — the pure function the routers and finalize share
  state.py       ResearchState + the reducers, each with its reasoning
  nodes/         plan, act, observe, compact, reflect, finalize
  tools/         registry + web_search, fetch_page, calculator, code_exec, file_ops
  trace.py       the canonical per-run trace schema
  demo_guard.py  daily spend cap, BYO-key, input limits for the public demo
evals/
  dataset.json   30 cases: 12 easy, 11 multi-hop, 7 adversarial
  metrics.py     bipartite matching, P/R/F1, LCS ordering, pass@1 / pass^k, Wilson CIs
  judge.py       binary rubric with worked examples
  agreement.py   Cohen's κ, Gwet's AC1, bootstrap CI, confusion matrix
  results/       committed traces and results — the audit trail behind every number
tests/           272 tests, no API key required
```

## Licence

MIT — see [LICENSE](LICENSE).
