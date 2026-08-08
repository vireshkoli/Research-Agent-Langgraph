"""Deployment entrypoint. Kept thin on purpose — the UI lives in the package.

Render (and a Hugging Face Space, if this ever moves there) both run this file.
`PORT` is supplied by the host; 7860 is the local and Spaces default.
"""

import os

from research_agent.config import settings
from research_agent.ui.render import build

if __name__ == "__main__":
    demo = build()
    demo.queue(max_size=settings().queue_max_size).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        # Gradio 6 moved app-level options here from Blocks(...).
        footer_links=["gradio"],
        show_error=True,
    )
