"""The study-planner agent.

Pipeline (each step is a traced tool, so the whole run shows up as a tree in LangSmith):

  discover_topics -> resolve_scope -> build_schedule
      -> for each week / topic:  retrieve_context -> draft_tasks -> verify
                                                  -> draft_questions -> verify -> revise (reflection)
      -> evaluate_plan

The LLM (if configured) only ever sees retrieved context. Every item it returns is verified by the
SafetyPolicy; items that fail are dropped, regenerated once with the failure reasons as feedback, and
finally back-filled from the extractive (cannot-hallucinate) generator.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Settings, default_min_score
from .evaluators import evaluate_plan
from .generators import ExtractiveGenerator, GenerationError
from .llm import LLMError
from .models import ContextChunk, Question, StudyPlan, Task, TopicInfo, WeekPlan
from .safety import SafetyPolicy
from .schedule import build_schedule, scale_minutes, split_minutes
from .textutils import normalize_key, tokenize
from .tracing import TRACING_AVAILABLE, traced
from .vectorstore import VectorStore

MAX_CONTEXT_CHARS = 7000
MIN_TOPIC_CHARS = 80


class PlannerError(Exception):
    """A user-facing error (bad input, empty store, nothing supported by the material)."""


@dataclass
class PlanRequest:
    course: str = "My Course"
    weeks: int = 4
    hours_per_week: float = 6.0
    weak_topics: List[str] = field(default_factory=list)
    focus_topics: List[str] = field(default_factory=list)
    questions_per_topic: int = 2
    start_date: Optional[str] = None
    run_id: str = ""

    def validate(self) -> None:
        if not 1 <= int(self.weeks) <= 52:
            raise PlannerError("weeks must be between 1 and 52")
        if not 0.5 <= float(self.hours_per_week) <= 80:
            raise PlannerError("hours per week must be between 0.5 and 80")
        if not 0 <= int(self.questions_per_topic) <= 10:
            raise PlannerError("questions per topic must be between 0 and 10")
        if self.start_date:
            try:
                date.fromisoformat(self.start_date)
            except ValueError as exc:
                raise PlannerError("start date must look like YYYY-MM-DD") from exc


class StudyPlannerAgent:
    def __init__(self, settings: Settings, store: VectorStore, generator=None,
                 fallback: Optional[ExtractiveGenerator] = None, policy: Optional[SafetyPolicy] = None):
        self.settings = settings
        self.store = store
        self.fallback = fallback or ExtractiveGenerator()
        self.generator = generator or self.fallback
        self.policy = policy or SafetyPolicy(
            settings.min_grounding, settings.min_question_grounding, settings.min_task_grounding
        )
        self.min_score = default_min_score(store.backend)
        self._warnings: List[str] = []
        self._seen_questions: set = set()

    # ------------------------------------------------------------------ tools
    @traced("discover_topics", run_type="retriever")
    def discover_topics(self) -> List[TopicInfo]:
        return self.store.topics()

    @traced("resolve_scope", run_type="tool")
    def resolve_names(self, names: Sequence[str], topics: Sequence[TopicInfo]) -> Tuple[List[str], List[str]]:
        """Map free-text topic names to topics that really exist. Unknown names are returned as unresolved."""
        by_lower = {t.name.lower(): t.name for t in topics}
        resolved: List[str] = []
        unresolved: List[str] = []
        for raw in names:
            name = raw.strip()
            if not name:
                continue
            hit = by_lower.get(name.lower())
            if hit is None:  # token-containment match
                query = set(tokenize(name))
                best, best_score = None, 0.0
                for t in topics:
                    cand = set(tokenize(t.name))
                    inter = len(query & cand)
                    if inter and (query <= cand or cand <= query):
                        score = inter / len(query | cand)
                        if score > best_score:
                            best, best_score = t.name, score
                hit = best
            if hit is None:  # semantic match, but only if clearly relevant (rule R2)
                found = self.store.search(name, k=1)
                if found and found[0].score >= self.min_score:
                    hit = found[0].chunk.topic
            if hit is None:
                unresolved.append(name)
            elif hit not in resolved:
                resolved.append(hit)
        return resolved, unresolved

    @traced("retrieve_context", run_type="retriever")
    def retrieve_context(self, topic: str) -> List[ContextChunk]:
        in_topic = self.store.search(topic, k=self.settings.top_k, topic=topic)
        hits = list(in_topic)
        if sum(len(h.chunk.text) for h in hits) < 250:  # very thin section: add clearly relevant neighbours
            have = {h.chunk.chunk_id for h in hits}
            for extra in self.store.search(topic, k=self.settings.top_k):
                if extra.chunk.chunk_id not in have and extra.score >= self.min_score:
                    hits.append(extra)
                    have.add(extra.chunk.chunk_id)
                if len(hits) >= 3:
                    break
        # respect the context budget, keeping the best-scoring chunks
        hits.sort(key=lambda h: -h.score)
        kept, total = [], 0
        for h in hits:
            if kept and total + len(h.chunk.text) > MAX_CONTEXT_CHARS:
                continue
            kept.append(h)
            total += len(h.chunk.text)
        kept.sort(key=lambda h: self.store.order.get(h.chunk.chunk_id, 0))
        if sum(len(h.chunk.text) for h in kept) < MIN_TOPIC_CHARS:
            return []
        return [ContextChunk(sid=f"S{i}", chunk=h.chunk, score=h.score) for i, h in enumerate(kept, start=1)]

    def _call(self, method: str, **kwargs):
        """Call the primary generator; on failure fall back to the extractive generator."""
        try:
            return getattr(self.generator, method)(**kwargs)
        except (GenerationError, LLMError) as exc:
            if self.generator is self.fallback:
                raise
            self.policy.report.llm_failures += 1
            self.policy.report.fallbacks_used += 1
            self._warnings.append(f"The language model failed ({exc}); used the offline extractive generator instead.")
            return getattr(self.fallback, method)(**kwargs)

    @traced("draft_tasks", run_type="chain")
    def draft_tasks(self, topic: str, ctx: List[ContextChunk], minutes: int, mode: str) -> List[Task]:
        def verified(raw_items) -> List[Task]:
            tasks: List[Task] = []
            for raw in raw_items:
                task, reason = self.policy.check_task(raw, ctx)
                if task is None:
                    self.policy.report.drop(reason)
                else:
                    tasks.append(task)
            return tasks

        tasks = verified(self._call("tasks", topic=topic, context=ctx, minutes=minutes, mode=mode))
        if not tasks and self.generator is not self.fallback:
            self.policy.report.fallbacks_used += 1
            self._warnings.append(f"No verified tasks from the model for '{topic}'; used extractive tasks.")
            tasks = verified(self.fallback.tasks(topic=topic, context=ctx, minutes=minutes, mode=mode))
        return tasks[:6]

    @traced("draft_questions", run_type="chain")
    def draft_questions(self, topic: str, ctx: List[ContextChunk], n: int, mode: str = "study") -> List[Question]:
        accepted: List[Question] = []
        seen = self._seen_questions  # shared across the whole plan: no repeated questions

        def absorb(raw_items) -> Tuple[int, List[str]]:
            added, reasons = 0, []
            for raw in raw_items:
                if len(accepted) >= n:
                    break
                question, reason = self.policy.check_question(raw, ctx)
                if question is None:
                    self.policy.report.drop(reason)
                    reasons.append(reason.replace("_", " "))
                    continue
                key = normalize_key(question.question)
                if key in seen:
                    continue
                seen.add(key)
                accepted.append(question)
                added += 1
            return added, reasons

        feedback: Optional[str] = None
        for attempt in range(self.settings.max_revisions + 1):
            if len(accepted) >= n:
                break
            raw = self._call("questions", topic=topic, context=ctx, n=n - len(accepted), mode=mode, feedback=feedback)
            _, reasons = absorb(raw)
            if attempt > 0:
                self.policy.report.revisions_used += 1
            if not reasons:
                break  # nothing was rejected: asking again would not help
            feedback = "Rejected items: " + "; ".join(sorted(set(reasons))) + ". Use only the context."
        if len(accepted) < n and self.generator is not self.fallback:
            self.policy.report.fallbacks_used += 1
            absorb(self.fallback.questions(topic=topic, context=ctx, n=n, mode=mode))
        if len(accepted) < n:
            self._warnings.append(
                f"Only {len(accepted)} of {n} verified practice question(s) could be produced for '{topic}' "
                "from the course material."
            )
        return accepted

    @traced("evaluate_plan", run_type="tool")
    def evaluate(self, plan: StudyPlan, expected_topics: Sequence[str]) -> Dict[str, float]:
        return evaluate_plan(plan, expected_topics)

    def practice(self, topic_query: str, n: int = 3) -> Tuple[str, List[Question], List[str]]:
        """Solved practice questions for one topic (used by the `practice` CLI command)."""
        self.policy.reset()
        self._warnings = []
        self._seen_questions = set()
        topics = self.discover_topics()
        resolved, unresolved = self.resolve_names([topic_query], topics)
        if not resolved:
            raise PlannerError(
                f"'{topic_query}' is not covered by the course material, so no questions were generated "
                "(the planner refuses to invent content)."
            )
        topic = resolved[0]
        ctx = self.retrieve_context(topic)
        if not ctx:
            raise PlannerError(f"Not enough course material about '{topic}'.")
        questions = self.draft_questions(topic, ctx, n)
        return topic, questions, list(self._warnings)

    # ------------------------------------------------------------------ main run
    @traced("study_plan_agent", run_type="chain", tags=["study-planner"])
    def run(self, request: PlanRequest) -> StudyPlan:
        request.validate()
        self.policy.reset()
        self._warnings = []
        self._seen_questions = set()
        self.policy.report.injection_lines_removed = int(self.store.meta.get("injection_lines_removed", 0))

        topics = self.discover_topics()
        if not topics:
            raise PlannerError("The vector store is empty. Run: python -m study_planner ingest")

        unsupported: List[str] = []
        selected = topics
        if request.focus_topics:
            names, unsupported = self.resolve_names(request.focus_topics, topics)
            for u in unsupported:
                self.policy.report.topics_refused.append(u)
                self._warnings.append(
                    f"Topic '{u}' is not covered by the course material, so it was left out instead of being invented."
                )
            if not names:
                raise PlannerError(
                    "None of the requested topics are covered by the course material: "
                    + ", ".join(request.focus_topics)
                )
            selected = [t for t in topics if t.name in set(names)]

        selected_names = [t.name for t in selected]
        weak: List[str] = []
        if request.weak_topics:
            weak_names, weak_unknown = self.resolve_names(request.weak_topics, topics)
            weak = [w for w in weak_names if w in selected_names]
            for w in weak_unknown:
                self._warnings.append(f"Weak topic '{w}' is not in the course material and was ignored.")

        weights = {t.name: max(1.0, t.n_chars / 500.0) * (1.5 if t.name in weak else 1.0) for t in selected}
        slots, schedule_warnings = build_schedule(selected_names, [weights[n] for n in selected_names],
                                                  request.weeks, weak)
        total_minutes = request.hours_per_week * 60.0
        start = date.fromisoformat(request.start_date) if request.start_date else None
        info = {t.name: t for t in topics}
        used_chunks: Dict[str, None] = {}
        week_plans: List[WeekPlan] = []

        for slot in slots:
            study_m, practice_m, review_m = split_minutes(total_minutes, slot.kind)
            slot_weight = sum(weights[t] for t in slot.topics) or 1.0
            wp = WeekPlan(week=slot.week, kind=slot.kind, label=self._label(slot.week, start),
                          topics=list(slot.topics), study_minutes=study_m, practice_minutes=practice_m,
                          review_minutes=review_m)
            for topic in slot.topics:
                ctx = self.retrieve_context(topic)
                if not ctx:
                    self.policy.report.topics_refused.append(topic)
                    self._warnings.append(f"Not enough course material for '{topic}'; it was skipped.")
                    wp.topics.remove(topic)
                    continue
                budget = max(5, int(round(study_m * weights[topic] / slot_weight)))
                tasks = self.draft_tasks(topic, ctx, budget, slot.kind)
                if not tasks:
                    self._warnings.append(f"No verified study tasks could be created for '{topic}'.")
                    wp.topics.remove(topic)
                    continue
                for task, minutes in zip(tasks, scale_minutes([t.minutes for t in tasks], budget)):
                    task.minutes = minutes
                    wp.tasks.append(task)
                n_q = request.questions_per_topic
                if slot.kind == "revision" and n_q > 0:
                    n_q += 1
                if n_q > 0:
                    wp.questions.extend(self.draft_questions(topic, ctx, n_q, slot.kind))
                for item in [*tasks, *wp.questions]:
                    for cid in item.citations:
                        used_chunks[cid] = None
            if not wp.tasks:
                wp.notes.append("No material could be verified for this week.")
            if slot.kind == "revision":
                wp.notes.append("Revision week: re-attempt the questions without notes, then check your answers.")
            heavy = [t for t in wp.topics if t in weak]
            if heavy:
                wp.notes.append("Extra time allocated to your weak topics: " + ", ".join(heavy) + ".")
            flagged = sum(1 for q in wp.questions if q.needs_review)
            if flagged:
                wp.notes.append(f"{flagged} question(s) are marked 'needs review': check their numbers before relying on them.")
            week_plans.append(wp)

        run_id = request.run_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        sources = {}
        for cid in used_chunks:
            chunk = self.store.chunks[self.store.order[cid]]
            sources[cid] = {"source": chunk.source, "heading": chunk.heading_path, "doc_title": chunk.doc_title}

        plan = StudyPlan(
            plan_id=f"plan-{now:%Y%m%d-%H%M%S}-{run_id[:6]}",
            run_id=run_id,
            created_at=now.isoformat(timespec="seconds"),
            course=request.course,
            weeks_requested=request.weeks,
            hours_per_week=request.hours_per_week,
            questions_per_topic=request.questions_per_topic,
            generator=getattr(self.generator, "name", "unknown"),
            embedder=f"{self.store.backend}:{self.store.meta.get('model', '')}",
            weeks=week_plans,
            sources=sources,
            warnings=[*schedule_warnings, *self._warnings],
            unsupported_topics=unsupported,
        )
        plan.safety = self.policy.report.to_dict()
        plan.auto_scores = self.evaluate(plan, selected_names)
        return plan

    @staticmethod
    def _label(week: int, start: Optional[date]) -> str:
        if start is None:
            return f"Week {week}"
        first = start + timedelta(days=7 * (week - 1))
        last = first + timedelta(days=6)
        return f"Week {week} ({first:%b %d} - {last:%b %d})"


def run_plan(agent: StudyPlannerAgent, request: PlanRequest) -> StudyPlan:
    """Run the agent as one LangSmith root run whose id we control (needed for human feedback)."""
    if not request.run_id:
        request.run_id = str(uuid.uuid4())
    if TRACING_AVAILABLE:
        return agent.run(request, langsmith_extra={"run_id": uuid.UUID(request.run_id)})
    return agent.run(request)
