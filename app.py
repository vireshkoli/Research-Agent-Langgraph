"""Deployment entrypoint. Kept thin on purpose — the UI lives in the package.

Runs the same way on the Hugging Face Space and in the local Docker image.
`PORT` is supplied by the host; 7860 is the local and Spaces default.

The `sys.path` line exists for Hugging Face Gradio Spaces specifically. They install
`requirements.txt` from a temporary mount rather than the app directory, so the
package cannot be installed with `-e .` there — a relative `.` resolves to the
wrong place and the build fails. Putting `src/` on the path makes the src-layout
package importable without an install. Docker and local `uv run` already have it
installed, and the insert is harmless for them.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from research_agent.config import settings  # noqa: E402
from research_agent.ui.render import build  # noqa: E402

# ZeroGPU refuses to start a Space that declares no @spaces.GPU function — the
# runtime error is literally "No @spaces.GPU function detected during startup".
# This agent is IO-bound and never needs a GPU, but ZeroGPU is the only free way
# to host a Gradio Space on a personal account since cpu-basic was gated behind
# PRO. So a placeholder is declared and never called. ZeroGPU allocates the GPU
# on *call*, not on decoration, so nothing is ever reserved and the app runs
# entirely on the Space's CPU side. `spaces` exists only on Hugging Face; the
# import guard keeps local and Docker runs unaffected.
try:
    import spaces  # type: ignore[import-not-found]
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU
    def _zerogpu_placeholder() -> None:
        """Satisfies ZeroGPU's startup check. Deliberately never invoked."""


if __name__ == "__main__":
    demo = build()
    demo.queue(max_size=settings().queue_max_size).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        # Gradio 6 moved app-level options here from Blocks(...).
        footer_links=["gradio"],
        show_error=True,
    )
