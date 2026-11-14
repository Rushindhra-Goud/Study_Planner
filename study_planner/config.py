"""Central configuration. Reads environment variables (and a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, encoding="utf-8-sig", override=True)
except Exception:  # pragma: no cover
    pass

_PLACEHOLDERS = ("paste", "your_key", "your-key", "xxxx", "changeme")


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is not None:
        value = value.strip().strip("\"'").strip()
    return value if value not in (None, "") else default


def _clean_key(value: Optional[str]) -> Optional[str]:
    """Ignore empty values and placeholder text such as PASTE_YOUR_KEY_HERE."""
    if not value:
        return None
    if any(token in value.lower() for token in _PLACEHOLDERS):
        return None
    return value


def _resolve_provider() -> str:
    provider = (_env("STUDY_PLANNER_LLM", "auto") or "auto").lower()
    if provider == "auto":
        if _env("ANTHROPIC_API_KEY"):
            return "anthropic"
        if _env("OPENAI_API_KEY"):
            return "openai"
        return "offline"
    return provider


def default_min_score(backend: str) -> float:
    """Minimum cosine similarity for a chunk to count as 'relevant'."""
    return 0.30 if backend == "openai" else 0.12


@dataclass
class Settings:
    data_dir: Path
    store_dir: Path
    output_dir: Path
    llm_provider: str
    llm_model: str
    embedding_backend: str
    embedding_model: str
    chunk_chars: int
    top_k: int
    min_grounding: float
    min_question_grounding: float
    min_task_grounding: float
    max_revisions: int
    langsmith_project: str
    langsmith_dataset: str
    langsmith_api_key: Optional[str]
    tracing_allowed: bool

    @property
    def langsmith_enabled(self) -> bool:
        return bool(self.langsmith_api_key) and self.tracing_allowed

    def with_overrides(self, **kwargs) -> "Settings":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)


def load_settings(**overrides) -> Settings:
    provider = _resolve_provider()
    if provider == "anthropic":
        default_model = "claude-sonnet-5-5"
    elif provider == "openai":
        default_model = "gpt-4o-mini"
    else:
        default_model = "offline-extractive"

    backend = (_env("STUDY_PLANNER_EMBEDDINGS", "tfidf") or "tfidf").lower()
    settings = Settings(
        data_dir=Path(_env("STUDY_PLANNER_DATA_DIR", str(PROJECT_ROOT / "data" / "course_materials"))),
        store_dir=Path(_env("STUDY_PLANNER_STORE_DIR", str(PROJECT_ROOT / "storage" / "vectorstore"))),
        output_dir=Path(_env("STUDY_PLANNER_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))),
        llm_provider=provider,
        llm_model=_env("STUDY_PLANNER_MODEL", default_model) or default_model,
        embedding_backend=backend,
        embedding_model=_env("STUDY_PLANNER_EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small",
        chunk_chars=int(_env("STUDY_PLANNER_CHUNK_CHARS", "900") or 900),
        top_k=int(_env("STUDY_PLANNER_TOP_K", "6") or 6),
        min_grounding=float(_env("STUDY_PLANNER_MIN_GROUNDING", "0.6") or 0.6),
        min_question_grounding=float(_env("STUDY_PLANNER_MIN_Q_GROUNDING", "0.5") or 0.5),
        min_task_grounding=float(_env("STUDY_PLANNER_MIN_TASK_GROUNDING", "0.3") or 0.3),
        max_revisions=int(_env("STUDY_PLANNER_MAX_REVISIONS", "1") or 1),
        langsmith_project=_env("LANGSMITH_PROJECT", _env("LANGCHAIN_PROJECT", "study-planner")) or "study-planner",
        langsmith_dataset=_env("STUDY_PLANNER_DATASET", "study-planner-golden-plans") or "study-planner-golden-plans",
        langsmith_api_key=_clean_key(_env("LANGSMITH_API_KEY", _env("LANGCHAIN_API_KEY"))),
        tracing_allowed=(_env("STUDY_PLANNER_TRACING", "true") or "true").lower() not in ("0", "false", "no"),
    )
    return settings.with_overrides(**overrides)
