"""The Gradio interface: question in, live trace and a cited answer out.

Built for Gradio 6, which removed things older tutorials still use. The traps that
actually bite here:

- `gr.Chatbot`'s tuple format is **gone**; messages are dicts with role/content.
  (This UI uses Markdown panes rather than a chatbot, but the same removal governs
  any component that took tuples.)
- App-level parameters (`theme`, `css`, `js`, `head`) moved from `Blocks(...)` to
  `launch(...)`.
- `show_api` became `footer_links`.
- `gr.Chatbot.allow_tags` now defaults to **True**, so model output containing
  `<html>` is rendered rather than escaped. `gr.Markdown` is the component used
  here and takes `sanitize_html` instead, which already defaults to True — it is
  passed explicitly anyway, because these panes render raw model output and the
  behaviour should be stated rather than inherited.

The version is exact-pinned in `requirements.txt`, because Spaces and Render both
rebuild on push and a silent minor bump is the most common way a portfolio demo
quietly dies.

Progress is streamed with a plain generator: Gradio sends only the diff of each
yield, so streaming a long trace costs almost nothing.
"""

from collections.abc import Iterator
from typing import Any

import gradio as gr

from research_agent import demo_guard
from research_agent.agent import stream
from research_agent.config import settings
from research_agent.trace import RunTrace

# Every example is a case from evals/dataset.json with a verified reference answer.
# An earlier list included "the largest publicly disclosed Mistral model", which was
# never in the dataset and was therefore never checked — the agent answered Mixtral
# 8x7B (46.7B) when Mixtral 8x22B is 141B. A demo must not showcase a question the
# evaluation has not confirmed the agent can answer.
EXAMPLES = [
    # easy-001, passes 3/3
    "Who won the 2024 Nobel Prize in Physics, and what contribution was cited?",
    # multi-001, passes 3/3 — two hops plus arithmetic
    "Which is larger: the number of parameters in the largest Llama 3.1 model, or in "
    "GPT-3? Give both figures and the difference in billions.",
    # adv-002 — relevance detection: web_search is the wrong tool here
    "What is 847 multiplied by 293?",
    # adv-001 — unbounded scope; the agent should decline rather than fabricate
    "List every peer-reviewed paper published in 2025 that cites 'Attention Is All You Need'.",
]

NODE_LABELS = {
    "plan": "Planning",
    "act": "Choosing tools",
    "observe": "Running tools",
    "compact": "Compacting memory",
    "reflect": "Checking coverage",
    "finalize": "Writing the answer",
}

INTRO = """\
# Tool-Using Research Agent

A ReAct agent on an explicit LangGraph state graph. It decomposes a question, calls
tools in a loop, and returns a **cited** answer — with hard budgets on steps, wall
clock and dollars, so it can never run away.

Budgets are the feature, not a safety net bolted on: when one trips the run returns
a partial answer with what it did establish, never an error.
"""


def _progress_markdown(events: list[str], trace: RunTrace | None) -> str:
    lines = ["### Trace", ""]
    lines += events or ["_waiting…_"]
    if trace:
        totals = trace.usage["totals"]
        lines += [
            "",
            "---",
            f"**{trace.usage['steps']} steps** · **${totals['cost_usd']:.5f}** · "
            f"**{trace.timings_ms['total'] / 1000:.1f}s** · "
            f"{totals['input_tokens']:,} in / {totals['output_tokens']:,} out · "
            f"{trace.usage['search_credits']} search credits",
        ]
        if trace.outcome.early_exit_reason:
            lines.append(
                f"\n> Stopped early: **{trace.outcome.early_exit_reason}** — "
                "the answer above is what was established before the budget ran out."
            )
        if trace.outcome.used_deterministic_finalize:
            lines.append("\n> The answer was assembled without an LLM (deterministic fallback).")
    return "\n".join(lines)


def _sources_markdown(trace: RunTrace | None) -> str:
    if not trace or not trace.sources:
        return ""
    lines = ["### Sources", ""]
    for source in trace.sources:
        marker = "**cited**  " if source["sid"] in trace.citations else ""
        lines.append(f"- {marker}`[{source['sid']}]` [{source['title'][:90]}]({source['url']})")
    return "\n".join(lines)


def _describe(node: str, update: dict[str, Any]) -> str:
    """One human line per node, from the delta that node returned."""
    label = NODE_LABELS.get(node, node)

    if node == "plan" and (plan := update.get("plan")):
        items = "".join(f"\n    - {q}" for q in plan)
        return f"- **{label}** — {len(plan)} sub-question(s):{items}"

    if node == "act":
        calls = update.get("pending_calls") or []
        if not calls:
            return f"- **{label}** — ready to answer"
        named = ", ".join(f"`{c['name']}`" for c in calls)
        return f"- **{label}** — calling {named}"

    if node == "observe":
        observations = [
            observation
            for step in update.get("scratchpad", [])
            for observation in step.get("observations", [])
        ]
        if not observations:
            return f"- **{label}** — nothing to run"
        parts = []
        for observation in observations:
            status = "ok" if observation["ok"] else "failed"
            preview = str(next(iter(observation["args"].values()), ""))[:60]
            parts.append(f"\n    - `{observation['tool']}` ({status}) — {preview}")
        return f"- **{label}**{''.join(parts)}"

    if node == "reflect":
        decision = update.get("reflect_decision", "?")
        gaps = update.get("open_gaps") or []
        suffix = f" — {len(gaps)} gap(s) remaining" if gaps else ""
        return f"- **{label}** — {decision}{suffix}"

    if node == "compact":
        return f"- **{label}** — folded older steps into a summary"

    if node == "finalize":
        return f"- **{label}**"

    return f"- **{label}**"


def answer(question: str, own_key: str, variant: str) -> Iterator[tuple[str, str, str]]:
    """Generator driving the three output panes: answer, trace, sources."""
    refusal = demo_guard.check(question, own_key)
    if refusal:
        yield refusal, "", ""
        return

    events: list[str] = []
    yield "", _progress_markdown(events, None), ""

    trace: RunTrace | None = None
    with demo_guard.borrowed_key(own_key):
        for kind, payload in stream(question.strip(), variant=variant):  # type: ignore[arg-type]
            if kind == "progress":
                events.append(_describe(payload["node"], payload["update"]))
                yield "", _progress_markdown(events, None), ""
            else:
                trace = payload

    assert trace is not None  # stream always ends with a done event
    if not own_key:
        demo_guard.record(trace.usage["totals"]["cost_usd"])

    yield trace.answer, _progress_markdown(events, trace), _sources_markdown(trace)


def build() -> gr.Blocks:
    config = settings()

    with gr.Blocks(title="Tool-Using Research Agent") as demo:
        gr.Markdown(INTRO)
        budget_note = gr.Markdown(demo_guard.status_line())

        with gr.Row():
            question = gr.Textbox(
                label="Research question",
                placeholder="Ask something that needs looking up…",
                lines=2,
                max_length=config.max_question_chars,
                scale=4,
            )
            submit = gr.Button("Research", variant="primary", scale=1)

        with gr.Accordion("Options", open=False):
            variant = gr.Radio(
                choices=["full", "baseline", "no_overrule"],
                value="full",
                label="Variant",
                info="baseline is one search and one answer, with no loop — the "
                "comparison the evaluation reports against.",
            )
            own_key = gr.Textbox(
                label="Your own OpenAI API key (optional)",
                placeholder="sk-…",
                type="password",
                info="Used for your request only and never stored. Supplying one "
                "bypasses the shared daily budget.",
            )

        gr.Examples(
            examples=[[e] for e in EXAMPLES],
            inputs=[question],
            label="Examples — the last two are adversarial",
        )

        # These panes render raw model output, so HTML sanitisation is stated
        # explicitly rather than relied on as a default that could change.
        answer_pane = gr.Markdown(label="Answer", sanitize_html=True)
        with gr.Row():
            trace_pane = gr.Markdown(sanitize_html=True)
            sources_pane = gr.Markdown(sanitize_html=True)

        for trigger in (submit.click, question.submit):
            trigger(
                answer,
                inputs=[question, own_key, variant],
                outputs=[answer_pane, trace_pane, sources_pane],
                # Bounds simultaneous LLM calls. Not a budget — see demo_guard.
                concurrency_limit=config.concurrency_limit,
            ).then(demo_guard.status_line, outputs=budget_note)

    return demo
