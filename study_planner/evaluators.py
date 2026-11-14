"""Automatic (programmatic) plan-quality scores in [0, 1]. Humans add their own ratings on top."""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Dict, Sequence

from .models import StudyPlan


def evaluate_plan(plan: StudyPlan, expected_topics: Sequence[str]) -> Dict[str, float]:
    tasks = plan.all_tasks
    questions = plan.all_questions
    items = [*tasks, *questions]

    groundedness = mean(i.support for i in items) if items else 0.0
    citation_coverage = (sum(1 for i in items if i.citations) / len(items)) if items else 0.0

    covered = {t for w in plan.weeks if w.tasks for t in w.topics}
    expected = list(expected_topics)
    topic_coverage = (len([t for t in expected if t in covered]) / len(expected)) if expected else 1.0

    loads = [w.study_minutes + w.practice_minutes + w.review_minutes for w in plan.weeks]
    if len(loads) > 1 and mean(loads) > 0:
        workload_balance = max(0.0, 1.0 - pstdev(loads) / mean(loads))
    else:
        workload_balance = 1.0 if loads else 0.0

    if plan.questions_per_topic <= 0:
        question_quality = 1.0
        verified_rate = 1.0
    elif questions:
        good = [q for q in questions if len(q.solution) >= 30 and q.citations and not q.needs_review]
        question_quality = len(good) / len(questions)
        verified_rate = sum(1 for q in questions if not q.needs_review) / len(questions)
    else:
        question_quality = 0.0
        verified_rate = 0.0

    overall = (
        0.30 * groundedness + 0.20 * citation_coverage + 0.20 * topic_coverage
        + 0.10 * workload_balance + 0.20 * question_quality
    )
    scores = {
        "groundedness": groundedness,
        "citation_coverage": citation_coverage,
        "topic_coverage": topic_coverage,
        "workload_balance": workload_balance,
        "question_quality": question_quality,
        "verified_rate": verified_rate,
        "overall_auto": overall,
    }
    return {k: round(float(v), 4) for k, v in scores.items()}
