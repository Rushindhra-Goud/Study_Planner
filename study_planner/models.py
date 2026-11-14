"""Plain dataclasses shared across the project."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    chunk_id: str
    source: str
    doc_title: str
    topic: str
    heading_path: str
    text: str

    def embed_text(self) -> str:
        return f"{self.doc_title} | {self.heading_path}\n{self.text}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


@dataclass
class ContextChunk:
    """A retrieved chunk labelled with a short id (S1, S2, ...) for citation."""

    sid: str
    chunk: Chunk
    score: float = 0.0


@dataclass
class TopicInfo:
    name: str
    doc_title: str
    source: str
    n_chunks: int
    n_chars: int


@dataclass
class Task:
    title: str
    description: str
    minutes: int
    citations: List[str] = field(default_factory=list)  # chunk ids
    support: float = 1.0


@dataclass
class Question:
    question: str
    solution: str
    difficulty: str = "medium"
    citations: List[str] = field(default_factory=list)  # chunk ids
    source_kind: str = "llm"  # worked_example | definition | llm
    support: float = 1.0
    needs_review: bool = False
    review_reason: str = ""
    approved: Optional[bool] = None  # set by the human reviewer


@dataclass
class WeekPlan:
    week: int
    kind: str  # study | revision
    label: str
    topics: List[str] = field(default_factory=list)
    study_minutes: int = 0
    practice_minutes: int = 0
    review_minutes: int = 0
    tasks: List[Task] = field(default_factory=list)
    questions: List[Question] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class StudyPlan:
    plan_id: str
    run_id: str
    created_at: str
    course: str
    weeks_requested: int
    hours_per_week: float
    questions_per_topic: int = 2
    generator: str = "offline-extractive"
    embedder: str = "tfidf"
    weeks: List[WeekPlan] = field(default_factory=list)
    sources: Dict[str, Dict[str, str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    unsupported_topics: List[str] = field(default_factory=list)
    safety: Dict[str, Any] = field(default_factory=dict)
    auto_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudyPlan":
        data = dict(data)
        weeks = []
        for w in data.pop("weeks", []):
            w = dict(w)
            w["tasks"] = [Task(**t) for t in w.get("tasks", [])]
            w["questions"] = [Question(**q) for q in w.get("questions", [])]
            weeks.append(WeekPlan(**w))
        return cls(weeks=weeks, **data)

    @property
    def all_questions(self) -> List[Question]:
        return [q for w in self.weeks for q in w.questions]

    @property
    def all_tasks(self) -> List[Task]:
        return [t for w in self.weeks for t in w.tasks]
