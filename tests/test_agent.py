import pytest

from study_planner.agent import PlanRequest, PlannerError, StudyPlannerAgent
from study_planner.evaluators import evaluate_plan
from study_planner.generators import ExtractiveGenerator
from study_planner.llm import LLMError
from study_planner.schedule import build_schedule, partition, scale_minutes, split_minutes

from .conftest import make_generator


def test_partition_and_schedule():
    groups = partition([1, 1, 1, 1, 1, 1], 3)
    assert [len(g) for g in groups] == [2, 2, 2]
    assert partition([5, 1, 1], 3) == [[0], [1], [2]]
    slots, warns = build_schedule(["a", "b", "c"], [1, 1, 1], 6)
    assert len(slots) <= 6 and slots[0].kind == "study"
    slots, warns = build_schedule(["a", "b"], [1, 1], 2)
    assert len(slots) == 2 and not warns
    assert sum(scale_minutes([10, 20, 30], 90)) == 90
    assert sum(split_minutes(360, "study")) == 360


def test_full_offline_plan(settings, store):
    agent = StudyPlannerAgent(settings, store)
    plan = agent.run(PlanRequest(course="ML", weeks=4, hours_per_week=6, weak_topics=["gradient descent"]))
    assert len(plan.weeks) == 4
    covered = {t for w in plan.weeks for t in w.topics}
    assert covered == {t.name for t in store.topics()}
    for w in plan.weeks:
        assert w.study_minutes + w.practice_minutes + w.review_minutes == 360
        assert sum(t.minutes for t in w.tasks) > 0
    for item in [*plan.all_tasks, *plan.all_questions]:
        assert item.citations and all(c in plan.sources for c in item.citations)
    assert any("weak" in n.lower() for w in plan.weeks for n in w.notes)
    assert plan.auto_scores["overall_auto"] > 0.8
    qs = [q.question for q in plan.all_questions]
    assert len(qs) == len(set(qs))


def test_week_budget_matches_task_minutes(settings, store):
    agent = StudyPlannerAgent(settings, store)
    plan = agent.run(PlanRequest(weeks=4, hours_per_week=10))
    for w in plan.weeks:
        assert abs(sum(t.minutes for t in w.tasks) - w.study_minutes) <= 3 * len(w.topics)


def test_more_weeks_than_topics_adds_revision(settings, store):
    plan = StudyPlannerAgent(settings, store).run(PlanRequest(weeks=16, hours_per_week=5))
    assert len(plan.weeks) <= 16
    assert any(w.kind == "revision" for w in plan.weeks)
    assert plan.warnings  # shortened-plan warning


def test_short_plan_groups_topics(settings, store):
    plan = StudyPlannerAgent(settings, store).run(PlanRequest(weeks=2, hours_per_week=8))
    assert len(plan.weeks) == 2 and sum(len(w.topics) for w in plan.weeks) == len(store.topics())


def test_unknown_topic_is_refused_not_invented(settings, store):
    agent = StudyPlannerAgent(settings, store)
    plan = agent.run(PlanRequest(weeks=2, focus_topics=["gradient descent", "kubernetes networking"]))
    assert plan.unsupported_topics == ["kubernetes networking"]
    assert "kubernetes" not in " ".join(t.description.lower() for t in plan.all_tasks)
    assert any("not covered" in w for w in plan.warnings)
    with pytest.raises(PlannerError):
        agent.run(PlanRequest(focus_topics=["kubernetes networking"]))
    with pytest.raises(PlannerError):
        agent.practice("quantum chromodynamics")


@pytest.mark.parametrize("kwargs", [{"weeks": 0}, {"weeks": 99}, {"hours_per_week": 0},
                                    {"questions_per_topic": -1}, {"start_date": "tomorrow"}])
def test_invalid_requests(settings, store, kwargs):
    with pytest.raises(PlannerError):
        StudyPlannerAgent(settings, store).run(PlanRequest(**kwargs))


def test_zero_questions(settings, store):
    plan = StudyPlannerAgent(settings, store).run(PlanRequest(weeks=2, questions_per_topic=0))
    assert plan.all_questions == [] and plan.auto_scores["question_quality"] == 1.0


def test_start_date_labels(settings, store):
    plan = StudyPlannerAgent(settings, store).run(PlanRequest(weeks=2, start_date="2026-11-16"))
    assert "Nov 16" in plan.weeks[0].label


def test_focus_topic_fuzzy_match(settings, store):
    plan = StudyPlannerAgent(settings, store).run(PlanRequest(weeks=1, focus_topics=["gradient descent"]))
    assert plan.weeks[0].topics == ["Gradient Descent"]


# ---------------------------------------------------------------- LLM paths with a fake model
GOOD_Q = {"questions": [{
    "question": "A model has weights 3 and -4 and lambda = 0.5. Compute the L2 penalty.",
    "solution": "The squared weights are 9 and 16 so the sum is 9 + 16 = 25. The penalty is 0.5 * 25 = 12.5.",
    "difficulty": "easy", "citations": ["S1"]}]}
GOOD_T = {"tasks": [{"title": "Study regularization", "description": "Read about the L2 penalty and ridge regression in the notes.",
                     "minutes": 30, "citations": ["S1"]}]}


class Router:
    """Fake LLM that answers task prompts and question prompts differently."""
    provider, model = "fake", "fake-1"

    def __init__(self, q_replies, t_reply=GOOD_T):
        import json
        self.q = list(q_replies)
        self.t = t_reply
        self.json = json
        self.q_calls = 0

    def complete(self, system, user, max_tokens=2048):
        if '"tasks"' in user.split("Return JSON")[-1]:
            r = self.t
        else:
            r = self.q[min(self.q_calls, len(self.q) - 1)]
            self.q_calls += 1
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else self.json.dumps(r)


def _agent(settings, store, router):
    from study_planner.generators import LLMGenerator
    return StudyPlannerAgent(settings, store, LLMGenerator(router))


def test_llm_good_output_is_used(settings, store):
    agent = _agent(settings, store, Router([GOOD_Q]))
    topic, qs, warns = agent.practice("regularization", 1)
    assert topic == "Regularization" and len(qs) == 1 and not qs[0].needs_review


def test_hallucinated_llm_output_is_dropped_and_backfilled(settings, store):
    bad = {"questions": [{"question": "Explain how blockchain consensus works in detail.",
                          "solution": "Miners compete to solve cryptographic puzzles and the winner appends a block.",
                          "citations": ["S1"]},
                         {"question": "What is ridge regression according to the course?",
                          "solution": "It cites a source that does not exist and invents facts about neural pruning.",
                          "citations": ["S99"]}]}
    agent = _agent(settings, store, Router([bad]))
    _, qs, _ = agent.practice("regularization", 2)
    assert qs, "extractive fallback should back-fill"
    assert all("blockchain" not in q.question.lower() for q in qs)
    assert agent.policy.report.total_dropped >= 2
    assert agent.policy.report.fallbacks_used >= 1


def test_llm_revision_loop_recovers(settings, store):
    bad = {"questions": [{"question": "Explain blockchain mining difficulty adjustments.",
                          "solution": "Difficulty retargets using hashing power estimates every two weeks.",
                          "citations": ["S1"]}]}
    agent = _agent(settings, store, Router([bad, GOOD_Q]))
    _, qs, _ = agent.practice("regularization", 1)
    assert len(qs) == 1 and "L2 penalty" in qs[0].question
    assert agent.policy.report.revisions_used == 1


def test_invalid_json_falls_back(settings, store):
    agent = _agent(settings, store, Router(["this is not json at all"]))
    _, qs, warns = agent.practice("regularization", 2)
    assert qs and agent.policy.report.llm_failures >= 1
    assert any("failed" in w for w in warns)


def test_llm_exception_falls_back_for_full_plan(settings, store):
    class Down:
        provider, model = "fake", "down"

        def complete(self, *a, **k):
            raise LLMError("service unavailable")

    from study_planner.generators import LLMGenerator
    plan = StudyPlannerAgent(settings, store, LLMGenerator(Down())).run(PlanRequest(weeks=2))
    assert plan.all_tasks and plan.all_questions
    assert plan.safety["llm_failures"] > 0
    assert any("failed" in w for w in plan.warnings)


def test_markdown_wrapped_json_is_accepted():
    from study_planner.generators import extract_json
    assert extract_json('Sure!\n```json\n{"a": [1]}\n```') == {"a": [1]}
    assert extract_json('Here: {"a": 2} thanks') == {"a": 2}


def test_evaluators_penalise_empty_plan(settings, store):
    plan = StudyPlannerAgent(settings, store).run(PlanRequest(weeks=2))
    for w in plan.weeks:
        w.tasks, w.questions = [], []
    scores = evaluate_plan(plan, [t.name for t in store.topics()])
    assert scores["topic_coverage"] == 0 and scores["overall_auto"] < 0.2
