#!/usr/bin/env bash
# Deploy this repo to a Hugging Face Space on ZeroGPU hardware.
#
# Why ZeroGPU for an agent that never touches a GPU: since mid-2026 Hugging Face
# gates *creation* of Gradio and Docker Spaces on cpu-basic behind a PRO
# subscription (the API returns HTTP 402). Free personal accounts in good standing
# may still host two ZeroGPU Spaces. ZeroGPU only allocates a GPU when a function
# is decorated with @spaces.GPU, and nothing here is, so the agent runs on the CPU
# side of a ZeroGPU Space for free. It is the one free route left.
#
# Constraints that follow from that choice:
#   - ZeroGPU is Gradio-SDK only, so the Space builds from requirements.txt and
#     app.py rather than the Dockerfile. (The Dockerfile stays for local parity.)
#   - ZeroGPU pins Python to 3.12 or 3.10; the frontmatter sets 3.12.
#   - The account must be older than ~30 days; before that the API says so.
#
# Prereqs: HF_TOKEN (write scope) exported, or present in .env.
#
# Usage:
#   ./scripts/deploy_hf.sh <hf-username> [space-name]
#   ./scripts/deploy_hf.sh vireshk research-agent --secrets   # also push API keys
#
# Secrets are pushed from .env only when --secrets is given; they are never
# written into the repo or the upload.
set -euo pipefail

USER="${1:?usage: deploy_hf.sh <hf-username> [space-name] [--secrets]}"
SPACE="${2:-research-agent}"
PUSH_SECRETS=0
for arg in "$@"; do [[ "$arg" == "--secrets" ]] && PUSH_SECRETS=1; done
REPO_ID="$USER/$SPACE"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Same .env fallback llm.py uses, so this works from a fresh shell.
if [[ -z "${HF_TOKEN:-}" && -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "$ROOT/.env" | cut -d= -f2- || true)"
fi
: "${HF_TOKEN:?export HF_TOKEN (a write token from https://hf.co/settings/tokens)}"
export HF_TOKEN

echo "==> Creating Space $REPO_ID (Gradio SDK, ZeroGPU) if it doesn't exist"
uv run python - "$REPO_ID" <<'PY'
import os, sys
from huggingface_hub import create_repo
from huggingface_hub.utils import HfHubHTTPError

repo_id = sys.argv[1]
try:
    create_repo(repo_id, repo_type="space", space_sdk="gradio",
                space_hardware="zero-a10g", token=os.environ["HF_TOKEN"], exist_ok=True)
except HfHubHTTPError as exc:
    code = exc.response.status_code if exc.response is not None else None
    if code == 402:
        sys.exit(
            "\nHugging Face returned 402 Payment Required.\n"
            "cpu-basic Gradio/Docker Spaces need PRO. ZeroGPU is the free route, but "
            "the account must be ~30 days old; if it is, retry, otherwise wait.\n"
        )
    raise
print("ok:", repo_id)
PY

echo "==> Regenerating requirements.txt from the lockfile"
( cd "$ROOT" && uv run python scripts/requirements.py >/dev/null )

echo "==> Uploading repo (frontmatter README swapped in; secrets, caches, data, tests excluded)"
uv run python - "$REPO_ID" "$ROOT" <<'PY'
import os, sys, tempfile, shutil, pathlib
from huggingface_hub import upload_folder

repo_id, root = sys.argv[1], pathlib.Path(sys.argv[2])
staging = pathlib.Path(tempfile.mkdtemp())

# Nothing that must not reach a public Space, and nothing the Space does not need
# to run: secrets, local state, caches, the eval corpus, tests, CI, the report.
ignore = shutil.ignore_patterns(
    ".git", ".venv", ".env", ".spend.json", ".cache", "data", "workspace",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__",
    "docs", "evals", "tests", ".github", "render.yaml", "docker-compose.yml",
)
# shutil.ignore_patterns only prunes *inside* a copytree call; it never sees the
# top-level entry itself. So the top level is filtered explicitly here, for
# directories and files alike. Anything secret, host-specific, or not needed
# to run the app stays behind — the first deploy attempt tried to upload the
# LLM cassette cache and Hugging Face rejected the '.cache/' path.
skip_top = {
    ".git", ".venv", ".env", ".spend.json", ".spend.json.tmp", ".cache", "data",
    "workspace", ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__",
    "docs", "evals", "tests", ".github", "render.yaml", "docker-compose.yml",
    ".DS_Store", ".pre-commit-config.yaml",
}
for item in root.iterdir():
    if item.name in skip_top:
        continue
    dest = staging / item.name
    if item.is_dir():
        shutil.copytree(item, dest, ignore=ignore)
    else:
        shutil.copy2(item, dest)

# HF Spaces reads the SDK, version and Python pin from README.md frontmatter.
shutil.copy2(root / "deploy" / "README_hf.md", staging / "README.md")

upload_folder(
    repo_id=repo_id, repo_type="space", folder_path=str(staging),
    token=os.environ["HF_TOKEN"], commit_message="Deploy Tool-Using Research Agent",
)
shutil.rmtree(staging)
print("uploaded to", repo_id)
PY

if [[ "$PUSH_SECRETS" == "1" ]]; then
  echo "==> Pushing OPENAI_API_KEY and TAVILY_API_KEY from .env as Space secrets"
  uv run python - "$REPO_ID" "$ROOT" <<'PY'
import os, sys, pathlib
from huggingface_hub import HfApi

repo_id, root = sys.argv[1], pathlib.Path(sys.argv[2])
env = {}
for line in (root / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

api = HfApi(token=os.environ["HF_TOKEN"])
for key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
    if env.get(key) and not env[key].endswith("..."):
        api.add_space_secret(repo_id, key, env[key])
        print("  set secret:", key)
    else:
        print("  skipped (not in .env):", key)
# Demo-only settings. The project ledger is a dev guard and would brick the
# demo once cumulative spend crossed it, so it is disabled here.
for key, value in {"RA_DAILY_CAP_USD": "0.25", "RA_SESSION_CAP_USD": "0.05",
                   "RA_MAX_PROJECT_COST_USD": "0", "RA_MAX_RUN_COST_USD": "0.05"}.items():
    api.add_space_variable(repo_id, key, value)
    print("  set variable:", key, "=", value)
PY
fi

echo "==> Done. Space: https://huggingface.co/spaces/$REPO_ID"
echo "    App URL:     https://$USER-$SPACE.hf.space"
[[ "$PUSH_SECRETS" == "1" ]] || echo "    Re-run with --secrets to push API keys, or set them in the Space UI."
