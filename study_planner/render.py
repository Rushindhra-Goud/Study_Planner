"""Render a StudyPlan as Markdown and persist plans as JSON + Markdown."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .models import StudyPlan


def plan_to_markdown(plan: StudyPlan) -> str:
    lines: List[str] = []
    add = lines.append
    add(f"# Study plan: {plan.course}")
    add("")
    add(f"*Plan ID `{plan.plan_id}` - {plan.weeks_requested} week(s) requested, {plan.hours_per_week:g} h/week - "
        f"generator: {plan.generator} - embeddings: {plan.embedder}*")
    add("")
    add("| Week | Type | Topics | Study | Practice | Review |")
    add("|---|---|---|---|---|---|")
    for w in plan.weeks:
        add(f"| {w.label} | {w.kind} | {', '.join(w.topics) or '-'} | {w.study_minutes} min | "
            f"{w.practice_minutes} min | {w.review_minutes} min |")
    add("")

    for w in plan.weeks:
        add(f"## {w.label} - {w.kind}")
        add("")
        add("**Topics:** " + (", ".join(w.topics) or "none"))
        add("")
        for note in w.notes:
            add(f"> {note}")
        if w.notes:
            add("")
        if w.tasks:
            add("### Study tasks")
            for t in w.tasks:
                cites = ", ".join(f"`{c}`" for c in t.citations)
                add(f"- **{t.title}** ({t.minutes} min): {t.description} [{cites}]")
            add("")
        if w.questions:
            add("### Practice questions (solved)")
            for i, q in enumerate(w.questions, start=1):
                flag = "  **[needs review]**" if q.needs_review else ""
                cites = ", ".join(f"`{c}`" for c in q.citations)
                add(f"**Q{i}. ({q.difficulty}){flag}** {q.question}")
                add("")
                add(f"<details><summary>Solution</summary>\n\n{q.solution}\n\n*Sources: {cites}*\n\n</details>")
                if q.needs_review and q.review_reason:
                    add("")
                    add(f"*Review note: {q.review_reason}*")
                add("")
    if plan.sources:
        add("## Sources")
        for cid, info in sorted(plan.sources.items()):
            add(f"- `{cid}` - {info.get('doc_title', '')} ({info.get('source', '')}) > {info.get('heading', '')}")
        add("")
    if plan.warnings:
        add("## Warnings")
        for warning in plan.warnings:
            add(f"- {warning}")
        add("")
    s = plan.safety or {}
    add("## Safety report")
    add(f"- Items checked: {s.get('items_checked', 0)}; dropped: {s.get('total_dropped', 0)} "
        f"{s.get('items_dropped', {})}; flagged for review: {s.get('items_flagged', 0)}")
    add(f"- Prompt-injection lines removed at ingest: {s.get('injection_lines_removed', 0)}; "
        f"URLs stripped: {s.get('urls_stripped', 0)}")
    add(f"- Topics refused (not in material): {s.get('topics_refused', [])}; revisions: {s.get('revisions_used', 0)}; "
        f"LLM failures: {s.get('llm_failures', 0)}; fallbacks: {s.get('fallbacks_used', 0)}")
    add("")
    if plan.auto_scores:
        add("## Automatic quality scores (0-1)")
        for key, value in plan.auto_scores.items():
            add(f"- {key}: {value}")
        add("")
    return "\n".join(lines)


def save_plan(plan: StudyPlan, output_dir: Path) -> Path:
    plans_dir = Path(output_dir) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    json_path = plans_dir / f"{plan.plan_id}.json"
    json_path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (plans_dir / f"{plan.plan_id}.md").write_text(plan_to_markdown(plan), encoding="utf-8")
    return json_path


def load_plan(output_dir: Path, plan_id: str) -> Optional[StudyPlan]:
    path = Path(output_dir) / "plans" / f"{plan_id}.json"
    if not path.exists():
        return None
    return StudyPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_plan_ids(output_dir: Path) -> List[str]:
    plans_dir = Path(output_dir) / "plans"
    if not plans_dir.exists():
        return []
    return sorted(p.stem for p in plans_dir.glob("plan-*.json"))
