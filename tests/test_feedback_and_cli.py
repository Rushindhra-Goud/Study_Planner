import io
import json
import sys

import pytest

from study_planner import cli
from study_planner.agent import PlanRequest, StudyPlannerAgent
from study_planner.feedback import (
    FeedbackStore,
    LangSmithBridge,
    Rating,
    collect_ratings_interactive,
    record_rating,
    review_flagged_questions,
    sync_unsynced,
    validate_scores,
)
from study_planner.render import load_plan, plan_to_markdown, save_plan

from .conftest import FakeLangSmithClient


@pytest.fixture()
def plan(settings, store):
    return StudyPlannerAgent(settings, store).run(PlanRequest(weeks=2, run_id="11111111-1111-1111-1111-111111111111"))


def test_validate_scores():
    assert validate_scores({"overall": 4, "accuracy": None}) == {"overall": 4}
    with pytest.raises(ValueError):
        validate_scores({"overall": 7})
    with pytest.raises(ValueError):
        validate_scores({"accuracy": 3})
    with pytest.raises(ValueError):
        validate_scores({"overall": 3, "bogus": 1})


def test_rating_goes_to_langsmith_and_dataset(settings, plan):
    fake = FakeLangSmithClient()
    store = FeedbackStore(settings.output_dir)
    bridge = LangSmithBridge(settings, client=fake)
    rating = Rating(plan.plan_id, plan.run_id, {"overall": 5, "accuracy": 4}, comment="good", reviewer="amy")
    status = record_rating(rating, plan, store, bridge)
    assert status["langsmith"] and status["dataset"]
    keys = {f["key"] for f in fake.feedback}
    assert keys == {"human_overall", "human_accuracy"}
    overall = next(f for f in fake.feedback if f["key"] == "human_overall")
    assert overall["score"] == 1.0 and overall["value"] == 5 and overall["run_id"] == plan.run_id
    assert fake.examples and fake.examples[0]["metadata"]["plan_id"] == plan.plan_id
    assert store.all()[0].synced


def test_low_rating_not_added_to_dataset(settings, plan):
    fake = FakeLangSmithClient()
    status = record_rating(Rating(plan.plan_id, plan.run_id, {"overall": 2}), plan,
                           FeedbackStore(settings.output_dir), LangSmithBridge(settings, client=fake))
    assert status["langsmith"] and not status["dataset"] and not fake.examples


def test_offline_failure_then_sync(settings, plan):
    store = FeedbackStore(settings.output_dir)
    broken = LangSmithBridge(settings, client=FakeLangSmithClient(fail=True))
    status = record_rating(Rating(plan.plan_id, plan.run_id, {"overall": 4}), plan, store, broken)
    assert status["saved_locally"] and not status["langsmith"] and "network down" in status["error"]
    assert store.summary()["unsynced"] == 1

    healthy_client = FakeLangSmithClient()
    result = sync_unsynced(store, LangSmithBridge(settings, client=healthy_client), lambda pid: plan)
    assert result == {"sent": 1, "failed": 0}
    assert store.summary()["unsynced"] == 0 and healthy_client.examples


def test_no_langsmith_configured_stores_locally(settings, plan):
    bridge = LangSmithBridge(settings)  # no key, no client
    assert not bridge.available
    status = record_rating(Rating(plan.plan_id, plan.run_id, {"overall": 3}), plan,
                           FeedbackStore(settings.output_dir), bridge)
    assert status == {"saved_locally": True, "langsmith": False, "dataset": False, "error": None}


def test_summary_and_corrupt_line(settings, plan):
    store = FeedbackStore(settings.output_dir)
    store.add(Rating(plan.plan_id, plan.run_id, {"overall": 2, "clarity": 4}))
    store.add(Rating(plan.plan_id, plan.run_id, {"overall": 4}))
    with open(store.path, "a") as fh:
        fh.write("{corrupt\n")
    s = store.summary()
    assert s["n_ratings"] == 2 and s["mean_overall"] == 3 and s["low_rated_plans"] == [plan.plan_id]


def test_interactive_rating_and_review():
    from study_planner.models import Question, StudyPlan, WeekPlan
    q1 = Question("Q one?", "answer one is long enough", needs_review=True, review_reason="numbers")
    q2 = Question("Q two?", "answer two is long enough", needs_review=True, review_reason="numbers")
    p = StudyPlan("plan-x", "run-x", "now", "C", 1, 5, weeks=[WeekPlan(1, "study", "Week 1", questions=[q1, q2])])
    answers = iter(["y", "n"])
    votes = review_flagged_questions(p, input_fn=lambda _: next(answers), print_fn=lambda *_: None)
    assert [v["approved"] for v in votes] == [True, False]
    assert [q.question for q in p.weeks[0].questions] == ["Q one?"]

    replies = iter(["abc", "9", "4", "5", "", "3", "nice"])
    rating = collect_ratings_interactive(p, "bob", input_fn=lambda _: next(replies),
                                         print_fn=lambda *_: None, question_votes=votes)
    assert rating.scores == {"overall": 4, "accuracy": 5, "clarity": 3}  # workload skipped
    assert rating.comment == "nice" and rating.question_votes == votes

    assert collect_ratings_interactive(p, input_fn=lambda _: "", print_fn=lambda *_: None) is None


def test_plan_persistence_and_markdown(settings, plan):
    path = save_plan(plan, settings.output_dir)
    assert path.exists()
    loaded = load_plan(settings.output_dir, plan.plan_id)
    assert loaded.plan_id == plan.plan_id and len(loaded.weeks) == len(plan.weeks)
    assert load_plan(settings.output_dir, "nope") is None
    md = plan_to_markdown(loaded)
    assert "Practice questions" in md and "Safety report" in md and "Sources" in md


# ---------------------------------------------------------------- CLI end to end
def run_cli(tmp_path, *argv):
    base = ["--store-dir", str(tmp_path / "s"), "--output-dir", str(tmp_path / "o")]
    return cli.main([argv[0], *base, *argv[1:]])


def test_cli_workflow(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setenv("STUDY_PLANNER_LLM", "offline")
    assert run_cli(tmp_path, "ingest") == 0
    assert run_cli(tmp_path, "topics") == 0
    assert run_cli(tmp_path, "plan", "--weeks", "3", "--no-review", "--weak", "boosting") == 0
    out = capsys.readouterr().out
    plan_id = [l for l in out.splitlines() if l.startswith("Plan plan-")][0].split()[1].rstrip(":")
    assert run_cli(tmp_path, "rate", plan_id, "--overall", "5", "--comment", "ok") == 0
    assert run_cli(tmp_path, "report") == 0
    assert "mean overall: 5" in capsys.readouterr().out
    assert run_cli(tmp_path, "practice", "cross validation", "-n", "2") == 0
    assert run_cli(tmp_path, "plans") == 0
    assert cli.main(["sync", "--store-dir", str(tmp_path / "s"), "--output-dir", str(tmp_path / "o")]) == 1


def test_cli_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("STUDY_PLANNER_LLM", "offline")
    assert run_cli(tmp_path, "plan", "--no-review") == 1  # no store yet
    assert "ingest" in capsys.readouterr().err
    run_cli(tmp_path, "ingest")
    assert run_cli(tmp_path, "rate", "plan-missing", "--overall", "3") == 1
    assert run_cli(tmp_path, "rate", "plan-missing", "--overall", "9") == 1
    assert run_cli(tmp_path, "plan", "--weeks", "0", "--no-review") == 1
    assert run_cli(tmp_path, "practice", "quantum computing") == 1
    assert cli.main(["ingest", "--data-dir", str(tmp_path / "empty"), "--store-dir", str(tmp_path / "s2")]) == 1


def test_plan_runs_without_langsmith_installed(monkeypatch, settings, store):
    from study_planner import agent as agent_module
    monkeypatch.setattr(agent_module, "TRACING_AVAILABLE", False)
    # run_plan must call agent.run without the langsmith-only keyword argument
    plan = agent_module.run_plan(agent_module.StudyPlannerAgent(settings, store), agent_module.PlanRequest(weeks=1))
    assert plan.run_id and len(plan.weeks) == 1


def test_placeholder_key_is_ignored(monkeypatch):
    from study_planner.config import load_settings
    monkeypatch.setenv("LANGSMITH_API_KEY", "PASTE_YOUR_KEY_HERE")
    assert load_settings().langsmith_api_key is None
    monkeypatch.setenv("LANGSMITH_API_KEY", ' "lsv2_pt_abc123" ')
    assert load_settings().langsmith_api_key == "lsv2_pt_abc123"
    assert load_settings().langsmith_enabled


def test_check_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "PASTE_YOUR_KEY_HERE")
    run_cli(tmp_path, "ingest")
    capsys.readouterr()
    assert run_cli(tmp_path, "check") == 1  # no usable key -> reports the problem
    out = capsys.readouterr().out
    assert "[ OK ] vector store built" in out and "[FAIL] LANGSMITH_API_KEY" in out
