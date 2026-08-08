# Single stage on purpose: there is nothing to compile, so a builder stage would
# add a layer and save nothing.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RA_DEMO_DB=/app/data/demo.sqlite3 \
    PORT=7860

WORKDIR /app

# Dependencies first so a source edit does not invalidate the install layer.
# pyproject.toml declares `license = { file = "LICENSE" }`, so LICENSE has to be
# present at install time or the build fails on metadata validation.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY app.py ./

# Runs unprivileged. code_execution is a resource guardrail rather than a security
# boundary (see tools/code_exec.py), so the container should not be root either.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/workspace \
    && chown -R app:app /app
USER app

EXPOSE 7860
CMD ["sh", "-c", "python app.py"]
