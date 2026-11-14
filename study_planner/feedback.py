"""Human-in-the-loop evaluation.

Humans rate each generated plan (1-5 on several dimensions) and approve/reject flagged questions.
* Ratings are ALWAYS saved locally (outputs/feedback/ratings.jsonl) so nothing is lost offline.
* If LangSmith is configured, they are also attached to the plan's root run as feedback
  (keys ``human_overall``, ``human_accuracy`` ...), and highly rated plans are added to a LangSmith
  dataset that can be used as golden examples for regression tests.
* ``sync`` pushes anything that could not be sent earlier.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

from .config import Settings
from .models import StudyPlan

RATING_DIMENSIONS = ("overall", "accuracy", "workload", "clarity")
DATASET_MIN_RATING = 4


@dataclass
class Rating:
    plan_id: str
    run_id: str
    scores: Dict[str, int]
    comment: str = ""
    reviewer: str = "anonymous"
    question_votes: List[Dict[str, Any]] = field(default_factory=list)
    rating_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    synced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rating":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})  # type: ignore[attr-defined]


def validate_scores(scores: Dict[str, Any]) -> Dict[str, int]:
    clean: Dict[str, int] = {}
    for key, value in scores.items():
        if value is None:
            continue
        if key not in RATING_DIMENSIONS:
            raise ValueError(f"unknown rating dimension {key!r}; use one of {', '.join(RATING_DIMENSIONS)}")
        number = int(value)
        if not 1 <= number <= 5:
            raise ValueError(f"{key} rating must be between 1 and 5 (got {number})")
        clean[key] = number
    if "overall" not in clean:
        raise ValueError("an overall rating (1-5) is required")
    return clean


# ----------------------------------------------------------------------------- local store
class FeedbackStore:
    def __init__(self, output_dir: Path):
        self.dir = Path(output_dir) / "feedback"
        self.path = self.dir / "ratings.jsonl"

    def add(self, rating: Rating) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rating.to_dict(), ensure_ascii=False) + "\n")

    def all(self) -> List[Rating]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(Rating.from_dict(json.loads(line)))
                except (ValueError, TypeError):
                    continue  # skip a corrupt line instead of failing
        return out

    def save_all(self, ratings: List[Rating]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            for r in ratings:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    def mark_synced(self, rating_id: str) -> None:
        ratings = self.all()
        for r in ratings:
            if r.rating_id == rating_id:
                r.synced = True
        self.save_all(ratings)

    def summary(self) -> Dict[str, Any]:
        ratings = self.all()
        out: Dict[str, Any] = {"n_ratings": len(ratings), "unsynced": sum(1 for r in ratings if not r.synced)}
        for dim in RATING_DIMENSIONS:
            values = [r.scores[dim] for r in ratings if dim in r.scores]
            out[f"mean_{dim}"] = round(mean(values), 2) if values else None
        out["low_rated_plans"] = sorted({r.plan_id for r in ratings if r.scores.get("overall", 5) <= 2})
        return out


# ----------------------------------------------------------------------------- LangSmith
class LangSmithBridge:
    """All LangSmith calls live here. Every method returns False instead of raising on failure."""

    def __init__(self, settings: Settings, client: Any = None):
        self.settings = settings
        self._client = client
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self._client is not None or self.settings.langsmith_enabled

    @property
    def client(self):
        if self._client is None:
            from langsmith import Client

            self._client = Client(api_key=self.settings.langsmith_api_key)
        return self._client

    def _fail(self, exc: Exception) -> bool:
        self.last_error = f"{type(exc).__name__}: {exc}"
        return False

    def check_connection(self) -> Optional[str]:
        """Return None if LangSmith accepts the key, otherwise a short error message."""
        try:
            list(self.client.list_projects(limit=1))
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {str(exc)[:300]}"

    def push_auto_scores(self, run_id: str, scores: Dict[str, float]) -> bool:
        if not self.available:
            return False
        try:
            for key, value in scores.items():
                self.client.create_feedback(
                    run_id, key=f"auto_{key}", score=float(value),
                    feedback_source_type="model", comment="programmatic plan evaluator",
                )
            return True
        except Exception as exc:
            return self._fail(exc)

    def push_rating(self, rating: Rating) -> bool:
        if not self.available:
            return False
        try:
            source = {"reviewer": rating.reviewer, "kind": "human", "rating_id": rating.rating_id}
            for dim, value in rating.scores.items():
                self.client.create_feedback(
                    rating.run_id,
                    key=f"human_{dim}",
                    score=(value - 1) / 4.0,  # normalised to 0..1, raw value kept below
                    value=value,
                    comment=rating.comment if dim == "overall" and rating.comment else None,
                    feedback_source_type="api",
                    source_info=source,
                )
            for vote in rating.question_votes:
                self.client.create_feedback(
                    rating.run_id,
                    key="human_flagged_question_approved",
                    score=1.0 if vote.get("approved") else 0.0,
                    comment=str(vote.get("question", ""))[:300],
                    feedback_source_type="api",
                    source_info=source,
                )
            return True
        except Exception as exc:
            return self._fail(exc)

    def add_to_dataset(self, plan: StudyPlan, rating: Rating) -> bool:
        """Store a highly rated plan as a golden example in a LangSmith dataset."""
        if not self.available:
            return False
        try:
            name = self.settings.langsmith_dataset
            client = self.client
            if client.has_dataset(dataset_name=name):
                dataset = client.read_dataset(dataset_name=name)
            else:
                dataset = client.create_dataset(
                    name, description="Human-approved study plans (rating >= 4) from the Study Planner."
                )
            client.create_example(
                inputs={
                    "course": plan.course,
                    "weeks": plan.weeks_requested,
                    "hours_per_week": plan.hours_per_week,
                    "questions_per_topic": plan.questions_per_topic,
                },
                outputs={"plan": plan.to_dict()},
                dataset_id=dataset.id,
                metadata={"plan_id": plan.plan_id, "overall_rating": rating.scores.get("overall"),
                          "reviewer": rating.reviewer},
            )
            return True
        except Exception as exc:
            return self._fail(exc)


def record_rating(rating: Rating, plan: Optional[StudyPlan], store: FeedbackStore, bridge: LangSmithBridge,
                  dataset_min_rating: int = DATASET_MIN_RATING) -> Dict[str, Any]:
    """Save locally, then try LangSmith. Returns a small status dict for the CLI to print."""
    store.add(rating)
    status: Dict[str, Any] = {"saved_locally": True, "langsmith": False, "dataset": False, "error": None}
    if bridge.available:
        status["langsmith"] = bridge.push_rating(rating)
        if status["langsmith"]:
            store.mark_synced(rating.rating_id)
            rating.synced = True
            if plan is not None and rating.scores.get("overall", 0) >= dataset_min_rating:
                status["dataset"] = bridge.add_to_dataset(plan, rating)
        status["error"] = bridge.last_error
    return status


def sync_unsynced(store: FeedbackStore, bridge: LangSmithBridge, load_plan: Callable[[str], Optional[StudyPlan]]) -> Dict[str, int]:
    sent = failed = 0
    for rating in store.all():
        if rating.synced:
            continue
        if bridge.push_rating(rating):
            store.mark_synced(rating.rating_id)
            plan = load_plan(rating.plan_id)
            if plan is not None and rating.scores.get("overall", 0) >= DATASET_MIN_RATING:
                bridge.add_to_dataset(plan, rating)
            sent += 1
        else:
            failed += 1
    return {"sent": sent, "failed": failed}


# ----------------------------------------------------------------------------- interactive HITL
def _ask_int(prompt: str, input_fn: Callable[[str], str], print_fn: Callable[[str], None],
             optional: bool = True) -> Optional[int]:
    for _ in range(5):
        raw = input_fn(prompt).strip()
        if not raw:
            if optional:
                return None
            print_fn("  A rating from 1 to 5 is required.")
            continue
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print_fn("  Please enter a whole number from 1 to 5.")
    return None


def review_flagged_questions(plan: StudyPlan, input_fn: Callable[[str], str] = input,
                             print_fn: Callable[[str], None] = print) -> List[Dict[str, Any]]:
    """Ask the human to approve/reject every question the safety layer flagged. Rejected ones are removed."""
    votes: List[Dict[str, Any]] = []
    flagged = [(w, q) for w in plan.weeks for q in w.questions if q.needs_review and q.approved is None]
    if not flagged:
        return votes
    print_fn(f"\n{len(flagged)} question(s) were flagged for review (numbers not traceable to the course material).")
    for week, q in flagged:
        print_fn(f"\n[{week.label}] {q.question}\n  Solution: {q.solution}\n  Why flagged: {q.review_reason}")
        answer = input_fn("  Keep this question? [y/n] ").strip().lower()
        q.approved = answer.startswith("y")
        votes.append({"question": q.question, "approved": q.approved})
    for week in plan.weeks:
        week.questions = [q for q in week.questions if q.approved is not False]
    return votes


def collect_ratings_interactive(plan: StudyPlan, reviewer: str = "anonymous",
                                input_fn: Callable[[str], str] = input,
                                print_fn: Callable[[str], None] = print,
                                question_votes: Optional[List[Dict[str, Any]]] = None) -> Optional[Rating]:
    print_fn("\nPlease rate this plan (1 = poor, 5 = excellent). Press Enter on an optional item to skip it.")
    overall = _ask_int("  Overall quality [1-5]: ", input_fn, print_fn, optional=True)
    if overall is None:
        print_fn("No overall rating given: nothing was saved.")
        return None
    scores = {"overall": overall}
    for dim, label in (("accuracy", "Accuracy / grounded in my materials"),
                       ("workload", "Realistic workload"),
                       ("clarity", "Clarity of tasks and questions")):
        value = _ask_int(f"  {label} [1-5, optional]: ", input_fn, print_fn, optional=True)
        if value is not None:
            scores[dim] = value
    comment = input_fn("  Comment (optional): ").strip()
    return Rating(plan_id=plan.plan_id, run_id=plan.run_id, scores=scores, comment=comment,
                  reviewer=reviewer, question_votes=question_votes or [])
