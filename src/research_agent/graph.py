"""The state graph: six nodes, two routers, all edges static.

    START -> plan -> act -> observe -> {route_after_observe}
                                        |-> finalize      (budget verdict)
                                        |-> compact -> reflect
                                        `-> reflect
             reflect -> {route_after_reflect}
                                        |-> act           (gaps remain)
                                        |-> plan          (plan is wrong)
                                        `-> finalize      (covered, or budget)
             finalize -> END

Built explicitly rather than with `create_react_agent` (which LangGraph deprecated
in 1.0 in favour of `langchain.agents.create_agent`) because the point of this
project is that the control flow and the eval surface are inspectable. A prebuilt
agent hides both.

Three structural choices:

**`act` always edges to `observe`**, even when the model emitted no tool calls. In
that case `observe` is a no-op that sets `act_requested_stop`. Keeping the edge
static costs one cheap super-step and keeps the diagram honest — a reader can trust
that the arrows are the whole story.

**No error node.** Tool failures become error observations plus a
`consecutive_tool_failures` counter that the router reads. An error node would add a
super-step and buy nothing.

**Routers are pure.** They cannot write state, so `early_exit_reason` is set by
`finalize` re-running the same `budget_verdict` the router used. See budget.py.
"""

from collections.abc import Callable
from typing import Any, Final, Literal

from langgraph.graph import END, START, StateGraph

from research_agent.budget import budget_verdict
from research_agent.config import Settings, settings
from research_agent.state import ResearchState

# Nodes return a *partial* state update. ResearchState is total=False, so a partial
# dict is a valid instance of it — which keeps the return type honest instead of
# widening it to dict[str, Any] and losing key checking inside the node bodies.
NodeFn = Callable[[ResearchState], ResearchState]

# Node names. Final makes mypy infer the literal type, so a router returning the
# wrong constant is a type error rather than a silently dead edge.
PLAN: Final = "plan"
ACT: Final = "act"
OBSERVE: Final = "observe"
COMPACT: Final = "compact"
REFLECT: Final = "reflect"
FINALIZE: Final = "finalize"


def _last_act_prompt_tokens(state: ResearchState) -> int:
    """Real prompt size of the most recent act call.

    Compaction triggers on this rather than on a character estimate because the
    number is already there — every call records its usage — and an estimate would
    be wrong in exactly the case that matters, a single huge tool observation.
    """
    for call in reversed(state.get("llm_calls", [])):
        if call.get("purpose") == "act":
            return int(call.get("input_tokens", 0))
    return 0


def route_after_observe(
    state: ResearchState, cfg: Settings | None = None
) -> Literal["compact", "reflect", "finalize"]:
    """Budget first, then compaction, then the coverage check."""
    cfg = cfg or settings()

    if budget_verdict(state, cfg) is not None:
        return FINALIZE

    # baseline answers after a single round of tool use — plan, one act (which may
    # emit several parallel calls), observe, answer. No iteration.
    #
    # no_overrule keeps reflect for guidance but makes act's decision to stop final.
    # It is named for what it removes: reflect's ability to overrule a proposed stop,
    # which is the specific behaviour the evaluation set out to test. Reflect still
    # runs when act wants to continue.
    if state.get("variant") == "baseline":
        return FINALIZE
    if state.get("variant") == "no_overrule":
        return FINALIZE if state.get("act_requested_stop") else REFLECT

    if (
        _last_act_prompt_tokens(state) > cfg.compact_threshold_tokens
        and state.get("compactions", 0) < cfg.max_compactions
    ):
        return COMPACT

    return REFLECT


def route_after_reflect(
    state: ResearchState, cfg: Settings | None = None
) -> Literal["act", "plan", "finalize"]:
    """Continue, replan, or answer.

    Budget is re-checked here because `reflect` itself costs money and time; a run
    can cross its cap between the two routers.
    """
    cfg = cfg or settings()

    if budget_verdict(state, cfg) is not None:
        return FINALIZE

    decision = state.get("reflect_decision")
    if decision == "replan":
        # max_replans is enforced by budget_verdict above, so an over-budget
        # replan has already been routed to finalize by the time we get here.
        return PLAN
    if decision == "continue":
        return ACT
    return FINALIZE


def build_graph(
    nodes: dict[str, NodeFn], cfg: Settings | None = None, checkpointer: Any = None
) -> Any:
    """Wire the graph from a mapping of node name to implementation.

    Taking the implementations as an argument is what lets the routing be tested
    exhaustively with stubs and zero API calls — the topology under test is the
    same object the real agent runs.

    `cfg` is bound into the routers here for the same reason the nodes are bound
    to a tracker: LangGraph invokes a conditional edge with state alone, so a
    router that reached for the global `settings()` would silently ignore any
    per-run override. That made `--max-steps 1` run two steps.
    """
    missing = {PLAN, ACT, OBSERVE, COMPACT, REFLECT, FINALIZE} - set(nodes)
    if missing:
        raise ValueError(f"build_graph is missing node(s): {sorted(missing)}")

    cfg = cfg or settings()
    builder = StateGraph(ResearchState)
    for name, fn in nodes.items():
        # add_node's overloads are written against LangGraph's own _Node protocols
        # and do not admit a plain Callable held in a variable, whatever its
        # signature. Passing the functions in a dict is what makes the routing
        # testable with stubs, so the dict stays and the overload is waived here
        # rather than the design bending around a type stub.
        builder.add_node(name, fn)  # type: ignore[call-overload]

    builder.add_edge(START, PLAN)
    builder.add_edge(PLAN, ACT)
    builder.add_edge(ACT, OBSERVE)
    builder.add_conditional_edges(
        OBSERVE,
        lambda state: route_after_observe(state, cfg),
        {COMPACT: COMPACT, REFLECT: REFLECT, FINALIZE: FINALIZE},
    )
    builder.add_edge(COMPACT, REFLECT)
    builder.add_conditional_edges(
        REFLECT,
        lambda state: route_after_reflect(state, cfg),
        {ACT: ACT, PLAN: PLAN, FINALIZE: FINALIZE},
    )
    builder.add_edge(FINALIZE, END)

    return builder.compile(checkpointer=checkpointer)


def mermaid() -> str:
    """The diagram in the README, generated from the same constants as the graph.

    Every node is declared with its label *before* any edge is drawn. Defining
    labels inline on first use is the more compact form, but it meant `finalize`
    and `reflect` were first referenced as bare ids by an earlier edge and only
    given labels further down — and GitHub's mermaid renderer failed the whole
    diagram with "Could not find a suitable point for the given distance", a
    layout error rather than a syntax one. Declaring first is the boring, robust
    form; a diagram that does not render is worth nothing.
    """
    return f"""flowchart TD
    S([START])
    {PLAN}["{PLAN} · decompose into sub-questions"]
    {ACT}["{ACT} · LLM emits tool calls"]
    {OBSERVE}["{OBSERVE} · run tools · mint source ids"]
    {COMPACT}["{COMPACT} · summarise old steps"]
    {REFLECT}["{REFLECT} · coverage check"]
    {FINALIZE}["{FINALIZE} · cited synthesis"]
    E([END])
    r1{{route_after_observe}}
    r2{{route_after_reflect}}

    S --> {PLAN}
    {PLAN} --> {ACT}
    {ACT} --> {OBSERVE}
    {OBSERVE} --> r1
    r1 -->|budget verdict| {FINALIZE}
    r1 -->|prompt over threshold| {COMPACT}
    r1 -->|otherwise| {REFLECT}
    {COMPACT} --> {REFLECT}
    {REFLECT} --> r2
    r2 -->|gaps remain| {ACT}
    r2 -->|plan is wrong| {PLAN}
    r2 -->|covered or budget| {FINALIZE}
    {FINALIZE} --> E"""
