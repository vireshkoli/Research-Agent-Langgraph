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
| **pass@1** — a single attempt succeeds | **87.8%** | [70–95%] |
| **pass^3** — *all three* attempts succeed | **76.7%** | [59–88%] |
| Tool recall | 100.0% | — |
| Tool precision | **40.4%** | — |
| Citation resolution (cited ids that exist) | 97.7% | — |
| Invented-URL rate | 0.0% | — |
| Cost per run | $0.0023 | p95 $0.0059 |
| Latency | 15.4s p50 | 79.7s p95 |

Full breakdown, per-tier results and failure modes: **[evals/REPORT.md](evals/REPORT.md)**.

### The headline finding is negative: the loop loses to one round of search

The point of an evaluation is to be able to be wrong, so here is the result that matters. A
**baseline** variant — plan once, issue every search in a single parallel round, answer — beats the
full iterative agent on **every tier**:

| Tier | Full agent | Baseline (single round) |
|---|---|---|
| easy | 100.0% | 100.0% |
| multi-hop | 78.8% | **84.8%** |
| adversarial | 81.0% | **100.0%** |
| **overall pass@1** | **87.8%** | **94.4%** |
| mean steps | 2.5 | 1.0 |
| mean cost | $0.0023 | $0.0008 |

The baseline is **3× cheaper, 2.5× shorter, and more accurate.** For this task distribution and this
model, iterative refinement is not just failing to pay for itself — it is actively destructive.

**Why — and it is not over-searching.** Tool recall is already 100% at the first round: the agent
finds everything it needs immediately, largely because a single `act` step emits four to six
parallel searches. Everything after that is re-litigating evidence it already had.

The obvious explanation would be that the loop keeps searching, and the traces say otherwise. Only
**2 of 305 searches** repeat an earlier query, and only 6% of tool calls return nothing new. What
the loop actually does is stop calling tools and keep going anyway: **77 of 228 rounds — 34% — ran
no tools at all.** `act` emits no calls when it believes it can answer, so each of those rounds is
the agent asking to stop, being overruled by `reflect`, and paying for an `act` call and a `reflect`
call to do nothing. One adversarial case spends eight consecutive rounds this way. The defect is a
stop condition that can be overruled without limit, not an agent that searches too much. On the
adversarial tier the effect is starkest — those cases reward recognising that you should stop, and
a variant that cannot help but stop scores 100% against the loop's 81%.

**The `reflect` node is implicated.** It exists to catch premature *stopping*, and it fired: 35
overrules across 90 runs. An ablation that makes act's stop decision final — `no_overrule` — scores
**91.1% against the full agent's 87.8%** over all 90 runs, with fewer steps. But that gap is
**+3.3 points with a 95% paired interval of [-4.4, +11.1] and McNemar p = 0.549**: directionally
consistent with the rest of the evidence and nowhere near significant on its own. An earlier version
of this README quoted 93.1% against 90.3% from a partial run that was missing the adversarial tier
and looked stronger than it was; the tier was re-run and the comparison is now complete. Reflect is
not exonerated — 34% of rounds are stops it overruled into nothing — but the ablation alone does not
carry the claim, and it is reported at the strength the data supports.

**Is the judge trustworthy?** This finding rests on judge verdicts, so the judge was checked
against 90 hand labels made before any verdict was visible: **Cohen's κ = 0.824** (raw agreement
96.7%, Gwet's AC1 0.959). More usefully than the headline number, the errors are one-directional —
the judge said CORRECT where the human said INCORRECT **zero times**, and all three disagreements
are it being *stricter* than the human. Every pass rate here is therefore a floor rather than a
flattering estimate, which is the direction an evaluation should fail in.

**Caveats, stated rather than buried.** At 11 multi-hop and 7 adversarial cases the per-tier
intervals are very wide, and the multi-hop difference (78.8% vs 84.8%) is well inside noise. The
adversarial gap and the overall direction are more robust, but this is 30 cases, not 300. The honest
claim is *"on this benchmark the loop did not earn its cost"*, not *"agent loops don't work"*.

`pass^3` is 76.7% against a pass@1 of 87.8%, and the gap between `pass^3` and `(pass@1)^3` is
**+0.090**. Because x^k is convex, that quantity is non-negative whenever per-case rates differ at
all, so a positive value means failures are **concentrated in a fixed set of cases** rather than
scattered randomly across runs. Those are specific hard cases to go and fix, not variance to average
away.

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

- **The loop does not earn its cost on this benchmark.** A single round of parallel search beats it
  on every tier, 3× cheaper. See the headline finding above. 34% of its rounds call no tools at all —
  the agent asking to stop and being overruled — and 11 of 90 runs hit a tool cap instead of stopping.
- **Tool precision understates the agent.** The headline 40.4% is matched/n_actual against
  `expected_tools`, which names at most one call for 81 of 90 runs while the agent is designed to
  fan out several parallel searches per round. A correct four-way fan-out against a one-call
  expectation scores 25% with nothing wasted. The number is a lower bound confounded with the
  dataset's own thinness, and `evals/REPORT.md` now measures redundancy directly instead of
  inferring it from precision. The order score is likewise computed on only the 3 runs that declare
  a `tool_order`.
- **The `no_overrule` ablation mixes two dates.** 20 of its runs were lost to a key rotation and
  re-run later, so its easy and multi-hop runs are from one day and its adversarial runs from
  another, with the web moving in between. The comparison is complete rather than truncated, but
  the two halves are not contemporaneous and `REPORT.md` says so inline.
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

1. **Make the loop earn its place, or remove it.** The evidence says a single parallel round wins.
   The loop's only defensible advantage is adaptive follow-up, so it should be gated on evidence that
   follow-up is needed — if a round adds no new sources, finish. Right now `reflect` is biased the
   other way and overrules correct stops.
2. **A larger labelled sample.** 90 items puts κ's lower bound at 0.580; a few hundred would
   tighten it enough to distinguish "almost perfect" from "moderate".
3. **A cross-family judge** (Claude or Gemini) to bound self-preference rather than disclaiming it.
4. **Grow the dataset to ~100 cases.** The interval width at n=30 is the binding constraint on saying
   anything is better than anything else.
5. **Make search results cheaper to read.** A rerank-and-truncate pass over search snippets before
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
