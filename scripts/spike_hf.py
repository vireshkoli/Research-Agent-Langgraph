"""Phase-1 spike: settle where the live demo can actually be hosted.

MEASURED RESULT (2026-07-30, free personal account, write-scoped token). Both free
paths are closed; the HF API rejects the create call outright:

  sdk=gradio, cpu-basic  -> "Static Spaces are free for everyone, but hosting Gradio
                             and Docker Spaces on free cpu-basic requires a PRO
                             subscription."
  sdk=gradio, zero-a10g  -> "You must be subscribed to PRO to host Spaces with
                             ZeroGPU. If you recently created your account, please
                             wait 30 days or request a community grant."
  sdk=static             -> created successfully (control: the token can create Spaces)

So the "free ZeroGPU Space that never calls @spaces.GPU" route does not exist — the
gate is on creation, before any GPU code would run. An existing Docker Space on this
account predates the change and is grandfathered; that is not a route for new work.

The ladder therefore resolves to: HF PRO ($9/mo) or Render free tier.

Read-only by default. Pass --create to attempt a Space and see the current error.

Run: `uv run python scripts/spike_hf.py`
"""

import os
import sys

import httpx

from research_agent.llm import _load_env_file

API = "https://huggingface.co/api"


def main() -> int:
    _load_env_file()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("  HF_TOKEN is not set. Create one at https://huggingface.co/settings/tokens")
        print("  (needs 'write' permission to create a Space)")
        return 1

    headers = {"Authorization": f"Bearer {token}"}

    response = httpx.get(f"{API}/whoami-v2", headers=headers, timeout=30.0)
    if response.status_code != 200:
        print(f"  FAIL: whoami returned {response.status_code} — token invalid or expired?")
        return 1
    me = response.json()

    name = me.get("name")
    is_pro = me.get("isPro", False)
    plan = me.get("plan") or ("pro" if is_pro else "free")
    token_role = (me.get("auth") or {}).get("accessToken", {}).get("role", "unknown")

    print(f"\n  account       {name}")
    print(f"  plan          {plan}")
    print(f"  isPro         {is_pro}")
    print(f"  token role    {token_role}")

    spaces = httpx.get(
        f"{API}/spaces", params={"author": name}, headers=headers, timeout=30.0
    ).json()
    print(f"  spaces owned  {len(spaces)}")
    for space in spaces:
        runtime = space.get("runtime") or {}
        print(
            f"    - {space.get('id')}  sdk={space.get('sdk')}  "
            f"hw={runtime.get('hardware', {}).get('current') or runtime.get('stage')}"
        )

    print("\n  verdict:")
    if is_pro:
        print("    PRO account -> CPU-basic Gradio Space available. Use it; an IO-bound")
        print("    agent has no use for a GPU.")
    else:
        print("    Free account -> BOTH free routes are closed (measured, see module docstring):")
        print("      cpu-basic Gradio/Docker : requires PRO")
        print("      ZeroGPU                 : requires PRO")
        print("    Deploy target is HF PRO ($9/mo) or Render free tier (750 hrs/mo,")
        print("    15-min spin-down, ~60s cold start).")

    if token_role != "write":
        print(f"\n  NOTE: token role is {token_role!r}; creating a Space needs 'write'.")

    if "--create" not in sys.argv:
        print("\n  Read-only run. Pass --create to attempt a Space and see the live error.")
        return 0

    response = httpx.post(
        f"{API}/repos/create",
        headers=headers,
        json={
            "type": "space",
            "name": "research-agent-langgraph",
            "private": False,
            "sdk": "gradio",
        },
        timeout=30.0,
    )
    print(f"\n  create gradio/cpu-basic -> HTTP {response.status_code}: {response.text[:300]}")
    return 0 if response.status_code < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
