"""The FRAMES sample, gold-source recall, and the answer-key exclusion. No network.

A synthetic benchmark file stands in for the real one, so the sampling arithmetic,
the drift check and the URL matching are all tested in CI without a download.
"""

import csv
import json
from pathlib import Path

import pytest
from evals.benchmarks import frames
from evals.schema import EvalCase


def _benchmark(tmp_path: Path, counts: dict[str, int]) -> Path:
    path = tmp_path / "test.tsv"
    fields = ["", "Prompt", "Answer", "reasoning_types", "wiki_links"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        index = 0
        for kind, count in counts.items():
            for _ in range(count):
                writer.writerow(
                    {
                        "": index,
                        "Prompt": f"question {index}",
                        "Answer": f"answer {index}",
                        "reasoning_types": f"{kind} | Temporal reasoning",
                        "wiki_links": "['https://en.wikipedia.org/wiki/Alpha']",
                    }
                )
                index += 1
    return path


@pytest.fixture
def benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = _benchmark(
        tmp_path, {"Multiple constraints": 50, "Numerical reasoning": 30, "Tabular": 20}
    )
    monkeypatch.setattr(frames, "CACHE", path)
    monkeypatch.setattr(frames, "download", lambda: path)
    return path


def test_the_sample_is_stratified_in_proportion_and_sums_exactly(benchmark: Path) -> None:
    sample = frames.draw(size=20, seed=1)
    assert sample["strata"] == {"Multiple constraints": 10, "Numerical reasoning": 6, "Tabular": 4}
    assert len(sample["ids"]) == 20 == len(set(sample["ids"]))


def test_largest_remainder_rounding_never_loses_or_gains_a_question(benchmark: Path) -> None:
    for size in (7, 13, 33, 61):
        assert sum(frames.draw(size=size, seed=3)["strata"].values()) == size


def test_the_same_seed_reproduces_the_sample_and_another_does_not(benchmark: Path) -> None:
    assert frames.draw(size=20, seed=5)["ids"] == frames.draw(size=20, seed=5)["ids"]
    assert frames.draw(size=20, seed=5)["ids"] != frames.draw(size=20, seed=6)["ids"]


def test_a_changed_benchmark_file_is_refused_rather_than_silently_resampled(
    benchmark: Path, tmp_path: Path
) -> None:
    committed = tmp_path / "sample.json"
    committed.write_text(json.dumps(frames.draw(size=10, seed=1)))
    assert len(frames.load(committed)) == 10

    benchmark.write_text(benchmark.read_text().replace("answer 3", "a different answer"))
    with pytest.raises(ValueError, match="has changed"):
        frames.load(committed)


def test_frames_cases_carry_their_gold_answer_and_sources(benchmark: Path, tmp_path: Path) -> None:
    committed = tmp_path / "sample.json"
    committed.write_text(json.dumps(frames.draw(size=5, seed=1)))
    case = frames.load(committed)[0]
    assert case.tier == "frames" and case.must_include == []
    assert case.gold_links == ["https://en.wikipedia.org/wiki/Alpha"]
    assert case.reference_answer.startswith("answer ")


@pytest.mark.parametrize(
    "found",
    [
        "https://en.wikipedia.org/wiki/James_Buchanan",
        "http://en.wikipedia.org/wiki/James_Buchanan",
        "https://en.m.wikipedia.org/wiki/James_Buchanan",
        "https://en.wikipedia.org/wiki/James_Buchanan#Early_life",
        "https://en.wikipedia.org/wiki/James%20Buchanan",
        "https://en.wikipedia.org/wiki/james_Buchanan",
    ],
)
def test_gold_links_match_across_the_ways_a_wikipedia_url_is_written(found: str) -> None:
    case = EvalCase(
        id="f",
        tier="frames",
        question="q",
        reference_answer="r",
        must_include=[],
        gold_links=["https://en.wikipedia.org/wiki/James_Buchanan"],
    )
    assert frames.gold_source_recall(case, [{"url": found}]) == 1.0


def test_gold_source_recall_is_the_share_of_gold_pages_found() -> None:
    case = EvalCase(
        id="f",
        tier="frames",
        question="q",
        reference_answer="r",
        must_include=[],
        gold_links=[f"https://en.wikipedia.org/wiki/{t}" for t in ("A", "B", "C")],
    )
    found = [{"url": "https://en.wikipedia.org/wiki/A"}, {"url": "https://en.wikipedia.org/wiki/C"}]
    assert frames.gold_source_recall(case, found) == pytest.approx(2 / 3)
    assert frames.gold_source_recall(case, []) == 0.0


def test_a_case_without_gold_sources_has_no_recall_rather_than_zero() -> None:
    case = EvalCase(id="h", tier="easy", question="q", reference_answer="r", must_include=["r"])
    assert frames.gold_source_recall(case, [{"url": "https://x"}]) is None


# --- the answer-key exclusion ------------------------------------------------------


@pytest.fixture
def exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_agent.config import settings

    monkeypatch.setenv("RA_SEARCH_EXCLUDE_DOMAINS", json.dumps(["huggingface.co", "github.com"]))
    settings.cache_clear()
    yield
    settings.cache_clear()


def test_search_drops_answer_key_hosts_and_the_unattributable_summary(exclusions: None) -> None:
    from research_agent.tools import web_search

    payload = {
        "answer": "Jane Ballou, according to the dataset",
        "results": [
            {"url": "https://huggingface.co/datasets/google/frames-benchmark", "title": "key"},
            {"url": "https://raw.github.com/x/frames", "title": "mirror"},
            {"url": "https://en.wikipedia.org/wiki/Harriet_Lane", "title": "Harriet Lane"},
        ],
    }
    result = web_search._render("q", payload, backend="ddgs")
    assert [s.url for s in result.sources] == ["https://en.wikipedia.org/wiki/Harriet_Lane"]
    assert "Jane Ballou" not in result.content
    assert result.meta["excluded"] == 2


def test_fetching_an_answer_key_host_is_refused(exclusions: None) -> None:
    from research_agent.tools.fetch_page import refusal

    assert "excluded" in (refusal("https://huggingface.co/datasets/google/frames-benchmark") or "")
    assert refusal("https://8.8.8.8/") is None


def test_with_no_exclusions_nothing_is_dropped() -> None:
    from research_agent.tools import web_search

    payload = {"answer": "summary", "results": [{"url": "https://github.com/a", "title": "a"}]}
    result = web_search._render("q", payload, backend="ddgs")
    assert len(result.sources) == 1 and "summary" in result.content


# --- the pre-registered analysis, exercised before any real data exists ------------


def _write_arm(root: Path, arm: str, outcomes: dict[str, tuple[bool, str | None]]) -> None:
    from dataclasses import asdict

    from evals.metrics import CaseResult, ToolMetrics

    rows = [
        asdict(
            CaseResult(
                case_id=case_id,
                tier="frames",
                run=1,
                success=success,
                judge_verdict="CORRECT" if success else "INCORRECT",
                judge_failure_mode=None,
                anchors_ok=True,
                forbidden_ok=True,
                behavior_ok=True,
                tools=ToolMetrics(1.0, 1.0, 1.0, 0, 0, 0, None),
                steps=2,
                step_efficiency=1.0,
                citation_resolution=1.0,
                invented_url_rate=None,
                cost_usd=0.003,
                latency_ms=9000.0,
                early_exit_reason=exit_reason,
                gold_source_recall=0.5,
            )
        )
        for case_id, (success, exit_reason) in outcomes.items()
    ]
    (root / "frames").mkdir(parents=True, exist_ok=True)
    (root / "frames" / f"{arm}.json").write_text(json.dumps({"results": rows}))


def test_the_registered_analysis_detects_a_clear_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import report

    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(frames, "SAMPLE", tmp_path / "absent.json")
    ids = [f"frames-{i:03d}" for i in range(60)]
    _write_arm(tmp_path, "full", {c: (True, None) for c in ids})
    _write_arm(tmp_path, "baseline", {c: (i % 2 == 0, None) for i, c in enumerate(ids)})

    text = "\n".join(report.frames_report())
    assert "**Significant.**" in text and "full agent scores higher by 50.0 points" in text
    assert "**60 questions**" in text


def test_the_registered_analysis_reports_a_bound_when_nothing_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import report

    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(frames, "SAMPLE", tmp_path / "absent.json")
    ids = [f"frames-{i:03d}" for i in range(40)]
    _write_arm(tmp_path, "full", {c: (i % 4 != 0, None) for i, c in enumerate(ids)})
    _write_arm(tmp_path, "baseline", {c: (i % 4 != 1, None) for i, c in enumerate(ids)})

    text = "\n".join(report.frames_report())
    assert "**No difference detected**" in text
    assert "does not mean the arms are equal" in text


def test_a_transport_failure_drops_the_question_from_both_arms_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import report

    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(frames, "SAMPLE", tmp_path / "absent.json")
    ids = [f"frames-{i:03d}" for i in range(10)]
    _write_arm(
        tmp_path, "full", {c: (True, "internal_error" if c == ids[3] else None) for c in ids}
    )
    _write_arm(tmp_path, "baseline", {c: (True, None) for c in ids})

    text = "\n".join(report.frames_report())
    assert "**9 questions**" in text
    assert f"**Excluded:** 1 ({ids[3]})" in text


def test_an_answer_key_source_in_a_trace_is_counted_as_contamination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import report

    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    traces = tmp_path / "frames" / "traces" / "full"
    traces.mkdir(parents=True)
    (traces / "frames-001-r1.json").write_text(
        json.dumps(
            {
                "sources": [
                    {"url": "https://huggingface.co/datasets/google/frames-benchmark/viewer"},
                    {"url": "https://en.wikipedia.org/wiki/A"},
                    {"url": "https://example.org/blog/frames-benchmark-answers"},
                ]
            }
        )
    )
    assert report._answer_key_hits(["frames-001"]) == 2


def test_nothing_is_reported_until_both_arms_have_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import report

    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    _write_arm(tmp_path, "full", {"frames-001": (True, None)})
    assert "Not run yet" in "\n".join(report.frames_report())
