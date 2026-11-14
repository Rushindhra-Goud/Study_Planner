"""Command line interface:  python -m study_planner <command> ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .agent import PlanRequest, PlannerError, StudyPlannerAgent, run_plan
from .config import ENV_FILE, PROJECT_ROOT, Settings, load_settings
from .embeddings import EmbeddingError
from .feedback import (
    FeedbackStore,
    LangSmithBridge,
    Rating,
    collect_ratings_interactive,
    record_rating,
    review_flagged_questions,
    sync_unsynced,
    validate_scores,
)
from .generators import ExtractiveGenerator, LLMGenerator
from .ingest import ingest_directory
from .llm import LLMError, make_llm
from .render import list_plan_ids, load_plan, plan_to_markdown, save_plan
from .tracing import flush_traces, setup_tracing
from .vectorstore import VectorStore, VectorStoreError


# ----------------------------------------------------------------------------- helpers
def _settings(args: argparse.Namespace) -> Settings:
    overrides = {
        "data_dir": Path(args.data_dir) if getattr(args, "data_dir", None) else None,
        "store_dir": Path(args.store_dir) if getattr(args, "store_dir", None) else None,
        "output_dir": Path(args.output_dir) if getattr(args, "output_dir", None) else None,
        "llm_provider": getattr(args, "llm", None),
        "llm_model": getattr(args, "model", None),
        "embedding_backend": getattr(args, "embeddings", None),
    }
    return load_settings(**overrides)


def _build_generator(settings: Settings):
    if settings.llm_provider == "offline":
        return ExtractiveGenerator()
    try:
        llm = make_llm(settings.llm_provider, settings.llm_model)
    except LLMError as exc:
        print(f"Warning: {exc}. Falling back to the offline extractive generator.", file=sys.stderr)
        return ExtractiveGenerator()
    return LLMGenerator(llm) if llm is not None else ExtractiveGenerator()


def _split(value: Optional[str]) -> List[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


# ----------------------------------------------------------------------------- commands
def cmd_ingest(args: argparse.Namespace) -> int:
    settings = _settings(args)
    result = ingest_directory(settings.data_dir, settings.chunk_chars)
    if not result.chunks:
        raise PlannerError(
            f"No usable course material found in {settings.data_dir} "
            "(supported: .md, .txt, .pdf). Skipped: " + ("; ".join(result.skipped) or "nothing")
        )
    notes = {
        "injection_lines_removed": len(result.removed_lines),
        "documents": result.documents,
        "course": Path(settings.data_dir).name,
    }
    store = VectorStore.build(result.chunks, settings.embedding_backend, settings.embedding_model, notes)
    store.save(settings.store_dir)
    print(f"Ingested {len(result.documents)} document(s) -> {len(result.chunks)} chunks, "
          f"{len(store.topics())} topics.")
    print(f"Embeddings: {store.backend} ({store.meta.get('model')}); store saved to {settings.store_dir}")
    for src, line in result.removed_lines:
        print(f"  [safety] removed suspicious line from {src}: {line[:80]!r}")
    for item in result.skipped:
        print(f"  [skipped] {item}")
    return 0


def cmd_topics(args: argparse.Namespace) -> int:
    store = VectorStore.load(_settings(args).store_dir)
    for i, t in enumerate(store.topics(), start=1):
        print(f"{i:2d}. {t.name}  ({t.doc_title}; {t.n_chunks} chunk(s))")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    settings = _settings(args)
    tracing = setup_tracing(settings)
    store = VectorStore.load(settings.store_dir)
    agent = StudyPlannerAgent(settings, store, _build_generator(settings))
    course = args.course or str(store.meta.get("course") or "My Course")
    request = PlanRequest(
        course=course, weeks=args.weeks, hours_per_week=args.hours,
        weak_topics=_split(args.weak), focus_topics=_split(args.topics),
        questions_per_topic=args.questions, start_date=args.start_date,
    )
    plan = run_plan(agent, request)
    flush_traces()
    bridge = LangSmithBridge(settings)
    if bridge.available:
        bridge.push_auto_scores(plan.run_id, plan.auto_scores)

    interactive = sys.stdin.isatty() and not args.no_review
    votes = review_flagged_questions(plan) if interactive else []
    path = save_plan(plan, settings.output_dir)

    if args.print:
        print(plan_to_markdown(plan))
    else:
        print(f"Plan {plan.plan_id}: {len(plan.weeks)} week(s), {len(plan.all_tasks)} task(s), "
              f"{len(plan.all_questions)} solved question(s).")
        for w in plan.weeks:
            print(f"  {w.label} [{w.kind}]: {', '.join(w.topics) or '-'}")
    for warning in plan.warnings:
        print(f"  warning: {warning}")
    print(f"Auto scores: {plan.auto_scores}")
    print(f"Saved: {path} (and .md)")
    print(f"LangSmith tracing: {'on, project ' + settings.langsmith_project if tracing else 'off (no LANGSMITH_API_KEY)'}")

    if interactive:
        rating = collect_ratings_interactive(plan, reviewer=args.reviewer, question_votes=votes)
        if rating is not None:
            status = record_rating(rating, plan, FeedbackStore(settings.output_dir), bridge)
            _print_rating_status(status)
    else:
        print(f"To rate this plan:  python -m study_planner rate {plan.plan_id} --overall 4 --comment \"...\"")
    return 0


def _print_rating_status(status: dict) -> None:
    print("Rating saved locally.")
    if status.get("langsmith"):
        print("Rating sent to LangSmith" + (" and plan added to the golden dataset." if status.get("dataset") else "."))
    elif status.get("error"):
        print(f"Could not reach LangSmith ({status['error']}). Run `sync` later to retry.")
    else:
        print("LangSmith is not configured (no key found), so the rating is stored locally only.\n"
              "Fix: put  LANGSMITH_API_KEY=lsv2_...  in the .env file, then run:  python -m study_planner check")


def cmd_practice(args: argparse.Namespace) -> int:
    settings = _settings(args)
    setup_tracing(settings)
    store = VectorStore.load(settings.store_dir)
    agent = StudyPlannerAgent(settings, store, _build_generator(settings))
    topic, questions, warnings = agent.practice(args.topic, args.n)
    print(f"Practice questions for: {topic}\n")
    for i, q in enumerate(questions, start=1):
        flag = " [needs review]" if q.needs_review else ""
        print(f"Q{i} ({q.difficulty}){flag}: {q.question}\n   Solution: {q.solution}\n   Sources: {', '.join(q.citations)}\n")
    for w in warnings:
        print(f"warning: {w}")
    return 0 if questions else 1


def cmd_rate(args: argparse.Namespace) -> int:
    settings = _settings(args)
    setup_tracing(settings)
    plan = load_plan(settings.output_dir, args.plan_id)
    if plan is None:
        raise PlannerError(f"Plan {args.plan_id!r} not found in {settings.output_dir / 'plans'}. "
                           "Use `plans` to list saved plans.")
    scores = validate_scores({"overall": args.overall, "accuracy": args.accuracy,
                              "workload": args.workload, "clarity": args.clarity})
    rating = Rating(plan_id=plan.plan_id, run_id=plan.run_id, scores=scores,
                    comment=args.comment or "", reviewer=args.reviewer)
    status = record_rating(rating, plan, FeedbackStore(settings.output_dir), LangSmithBridge(settings))
    _print_rating_status(status)
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    settings = _settings(args)
    setup_tracing(settings)
    bridge = LangSmithBridge(settings)
    if not bridge.available:
        raise PlannerError("LangSmith is not configured. Set LANGSMITH_API_KEY in .env first.")
    result = sync_unsynced(FeedbackStore(settings.output_dir), bridge,
                           lambda pid: load_plan(settings.output_dir, pid))
    print(f"Synced {result['sent']} rating(s); {result['failed']} failed." +
          (f" Last error: {bridge.last_error}" if result["failed"] else ""))
    return 0 if result["failed"] == 0 else 1


def cmd_report(args: argparse.Namespace) -> int:
    settings = _settings(args)
    summary = FeedbackStore(settings.output_dir).summary()
    print(f"Ratings collected: {summary['n_ratings']} (not yet in LangSmith: {summary['unsynced']})")
    for dim in ("overall", "accuracy", "workload", "clarity"):
        value = summary[f"mean_{dim}"]
        print(f"  mean {dim}: {value if value is not None else 'n/a'}")
    if summary["low_rated_plans"]:
        print("  plans rated 2 or lower: " + ", ".join(summary["low_rated_plans"]))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Self-diagnosis: tells you exactly what is missing instead of failing later."""
    import importlib.util

    ok = True

    def line(good: bool, text: str, fix: str = "") -> None:
        nonlocal ok
        print(("  [ OK ] " if good else "  [FAIL] ") + text + ("" if good or not fix else f"\n         -> {fix}"))
        ok = ok and good

    settings = _settings(args)
    print(f"Python {sys.version.split()[0]}")
    for module in ("numpy", "dotenv"):
        line(importlib.util.find_spec(module) is not None, f"package '{module}' installed",
             "pip install -r requirements.txt")
    line((PROJECT_ROOT / "data" / "course_materials").exists() or settings.data_dir.exists(), "course materials folder found")
    line(VectorStore.exists(settings.store_dir), "vector store built", "python -m study_planner ingest")

    print("\nLangSmith (optional, needed for uploading ratings):")
    if (PROJECT_ROOT / ".env.txt").exists():
        line(False, ".env.txt found", "rename it to exactly  .env  (turn on 'File name extensions' in Explorer)")
    line(ENV_FILE.exists(), f".env file found at {ENV_FILE}", "copy .env.example to .env in this folder")
    line(bool(settings.langsmith_api_key), "LANGSMITH_API_KEY set in .env",
         "open .env and paste your key after LANGSMITH_API_KEY= (no quotes, no spaces), then save")
    has_pkg = importlib.util.find_spec("langsmith") is not None
    line(has_pkg, "package 'langsmith' installed", "pip install -r requirements.txt")
    if settings.langsmith_api_key and has_pkg:
        error = LangSmithBridge(settings).check_connection()
        line(error is None, "LangSmith accepted the key and is reachable",
             f"{error}  (check the key was copied fully and that you are online)" if error else "")
    print("\n" + ("All good. Run:  python -m study_planner plan" if ok else "Fix the [FAIL] items above, then run check again."))
    return 0 if ok else 1


def cmd_plans(args: argparse.Namespace) -> int:
    settings = _settings(args)
    ids = list_plan_ids(settings.output_dir)
    if not ids:
        print("No saved plans yet.")
    for pid in ids:
        print(pid)
    return 0


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", help="folder with course materials (.md/.txt/.pdf)")
    common.add_argument("--store-dir", help="where the vector store is saved")
    common.add_argument("--output-dir", help="where plans and ratings are saved")

    llm_opts = argparse.ArgumentParser(add_help=False)
    llm_opts.add_argument("--llm", choices=["offline", "anthropic", "openai"], help="override the LLM provider")
    llm_opts.add_argument("--model", help="override the LLM model name")

    parser = argparse.ArgumentParser(prog="study_planner", description="Personalized study planner (RAG + agent + HITL).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", parents=[common], help="read course materials and build the vector store")
    p.add_argument("--embeddings", choices=["tfidf", "openai"], help="embedding backend")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("topics", parents=[common], help="list the topics found in the course materials")
    p.set_defaults(func=cmd_topics)

    p = sub.add_parser("plan", parents=[common, llm_opts], help="generate a weekly study plan with solved questions")
    p.add_argument("--course", help="course name shown in the plan")
    p.add_argument("--weeks", type=int, default=4, help="number of weeks (1-52)")
    p.add_argument("--hours", type=float, default=6.0, help="study hours per week")
    p.add_argument("--weak", help="comma-separated weak topics that need extra time")
    p.add_argument("--topics", help="comma-separated topics to restrict the plan to")
    p.add_argument("--questions", type=int, default=2, help="practice questions per topic (0-10)")
    p.add_argument("--start-date", help="first Monday, YYYY-MM-DD (adds dates to week labels)")
    p.add_argument("--no-review", action="store_true", help="skip the interactive human review/rating step")
    p.add_argument("--reviewer", default="anonymous", help="name stored with your rating")
    p.add_argument("--print", action="store_true", help="print the full plan as Markdown")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("practice", parents=[common, llm_opts], help="solved practice questions for one topic")
    p.add_argument("topic", help="topic name or description")
    p.add_argument("-n", type=int, default=3, help="number of questions")
    p.set_defaults(func=cmd_practice)

    p = sub.add_parser("rate", parents=[common], help="rate a saved plan (human feedback -> LangSmith)")
    p.add_argument("plan_id")
    p.add_argument("--overall", type=int, required=True, help="1-5")
    p.add_argument("--accuracy", type=int, help="1-5")
    p.add_argument("--workload", type=int, help="1-5")
    p.add_argument("--clarity", type=int, help="1-5")
    p.add_argument("--comment", help="free-text comment")
    p.add_argument("--reviewer", default="anonymous")
    p.set_defaults(func=cmd_rate)

    p = sub.add_parser("sync", parents=[common], help="send ratings that are not yet in LangSmith")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("report", parents=[common], help="summary of collected human ratings")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("check", parents=[common], help="diagnose setup problems (packages, .env, LangSmith key)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("plans", parents=[common], help="list saved plans")
    p.set_defaults(func=cmd_plans)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (PlannerError, VectorStoreError, EmbeddingError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return 130
