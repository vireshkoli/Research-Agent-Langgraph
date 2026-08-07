"""Cassette record/replay — the mechanism that keeps development affordable.

The claim is "the same prompt costs money once". These tests check that by counting
real client calls, not by trusting the cache to be wired up.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from research_agent import cassette, llm, spend


class FakeClient:
    """Counts calls so a test can prove the second one never reached the API."""

    def __init__(self) -> None:
        self.calls = 0
        self.responses = SimpleNamespace(create=self._create, parse=self._parse)

    def _usage(self) -> SimpleNamespace:
        return SimpleNamespace(
            input_tokens=1_000_000,
            output_tokens=0,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(output_text="the answer", output=[], usage=self._usage())

    def _parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        schema = kwargs["text_format"]
        return SimpleNamespace(output_parsed=schema(value="parsed"), output=[], usage=self._usage())


class Payload(llm.BaseModel):
    value: str


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient()
    monkeypatch.setattr(llm, "_client", lambda: fake)
    return fake


def test_cache_is_off_by_default() -> None:
    # Nothing should ever replay unless it was asked for. An official eval run that
    # silently replayed would make pass^k meaningless.
    assert cassette.mode() == "off"


def test_second_identical_call_is_free(client: FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RA_LLM_CACHE", "auto")

    first = llm.complete("sys", "user", model="gpt-5.4-nano", purpose="act")
    spent_after_first = spend.read().total_usd
    second = llm.complete("sys", "user", model="gpt-5.4-nano", purpose="act")

    assert first == second == "the answer"
    assert client.calls == 1, "the second call should have been replayed"
    assert spent_after_first == pytest.approx(0.20)  # 1M input tokens at $0.20/M
    assert spend.read().total_usd == pytest.approx(0.20), "replay must not add spend"


def test_a_different_prompt_is_a_different_recording(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RA_LLM_CACHE", "auto")
    llm.complete("sys", "question one", model="gpt-5.4-nano")
    llm.complete("sys", "question two", model="gpt-5.4-nano")
    assert client.calls == 2


def test_changing_the_model_busts_the_cache(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Replaying a nano response for a mini request would silently invalidate any
    # model comparison in the eval.
    monkeypatch.setenv("RA_LLM_CACHE", "auto")
    llm.complete("sys", "user", model="gpt-5.4-nano")
    llm.complete("sys", "user", model="gpt-5.4-mini")
    assert client.calls == 2


def test_changing_reasoning_effort_busts_the_cache(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RA_LLM_CACHE", "auto")
    llm.complete("sys", "user", model="gpt-5.4-nano", reasoning_effort=None)
    llm.complete("sys", "user", model="gpt-5.4-nano", reasoning_effort="low")
    assert client.calls == 2


def test_structured_output_round_trips_through_a_cassette(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RA_LLM_CACHE", "auto")
    first = llm.parse("sys", "user", Payload, model="gpt-5.4-nano")
    second = llm.parse("sys", "user", Payload, model="gpt-5.4-nano")

    assert client.calls == 1
    assert isinstance(second, Payload)
    assert first == second


def test_strict_replay_mode_fails_loudly_on_a_miss(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In strict replay a miss means the recording is stale. Falling through to a
    # paid call would defeat the point of asking for strict replay.
    monkeypatch.setenv("RA_LLM_CACHE", "replay")
    with pytest.raises(llm.CassetteMiss, match="RA_LLM_CACHE=replay"):
        llm.complete("sys", "never recorded", model="gpt-5.4-nano")
    assert client.calls == 0


def test_off_mode_never_replays(client: FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RA_LLM_CACHE", "off")
    llm.complete("sys", "user", model="gpt-5.4-nano")
    llm.complete("sys", "user", model="gpt-5.4-nano")
    assert client.calls == 2
    assert spend.read().total_usd == pytest.approx(0.40)


def test_an_unknown_mode_falls_back_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RA_LLM_CACHE", "yes-please")
    assert cassette.mode() == "off"
