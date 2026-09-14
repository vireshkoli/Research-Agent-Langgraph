---
title: Tool-Using Research Agent
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Tool-Using Research Agent

A ReAct agent on an explicit LangGraph state graph: it decomposes a question, calls
tools in a loop, and returns a **cited** answer — with hard budgets on steps, wall
clock and dollars, so it can never run away. When a budget trips, the run returns a
partial answer with what it did establish rather than an error.

Source, architecture, and the full measured evaluation (90 committed traces, a judge
validated against 90 hand labels, and a headline finding that a single-round baseline
beat the loop):
**https://github.com/vireshkoli/Research-Agent-Langgraph**

This Space runs on ZeroGPU hardware but never requests the GPU — the agent is
IO-bound and runs entirely on the CPU side. Spend is capped per visitor and per day;
paste your own OpenAI key in the app to bypass the cap.
