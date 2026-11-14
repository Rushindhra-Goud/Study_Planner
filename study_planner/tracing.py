"""LangSmith tracing helpers.

`traced(...)` wraps `langsmith.traceable` so that
  * it is a harmless no-op when tracing is disabled or langsmith is not installed;
  * inputs/outputs are converted to small JSON-safe values (no giant numpy arrays or objects).
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Dict, List, Optional

try:
    from langsmith import traceable as _traceable
except Exception:  # pragma: no cover - langsmith missing
    _traceable = None

TRACING_AVAILABLE = _traceable is not None

_MAX_STR = 2000
_MAX_ITEMS = 40


def _safe(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "<max depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + "...[truncated]"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _safe(dataclasses.asdict(value), depth + 1)
        except Exception:
            return repr(value)[:200]
    if isinstance(value, dict):
        return {str(k): _safe(v, depth + 1) for k, v in list(value.items())[:_MAX_ITEMS]}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out: List[Any] = [_safe(v, depth + 1) for v in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            out.append(f"...+{len(items) - _MAX_ITEMS} more")
        return out
    return repr(value)[:200]


def _process_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    safe = _safe(inputs)
    return safe if isinstance(safe, dict) else {"input": safe}


def _process_outputs(outputs: Any) -> Dict[str, Any]:
    safe = _safe(outputs)
    return safe if isinstance(safe, dict) else {"output": safe}


def traced(name: str, run_type: str = "chain", tags: Optional[List[str]] = None) -> Callable:
    def decorator(fn: Callable) -> Callable:
        if _traceable is None:
            return fn
        return _traceable(
            name=name,
            run_type=run_type,
            tags=tags or [],
            process_inputs=_process_inputs,
            process_outputs=_process_outputs,
        )(fn)

    return decorator


def setup_tracing(settings) -> bool:
    """Turn LangSmith tracing on only when an API key is configured. Returns True if enabled."""
    try:  # langsmith caches env lookups; clear so changes made here take effect
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()
    except Exception:
        pass
    if _traceable is not None and settings.langsmith_enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key or ""
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        enabled = True
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        enabled = False
    try:
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()
    except Exception:
        pass
    return enabled


def flush_traces() -> None:
    """Block until queued traces have been sent (call before pushing feedback to a fresh run)."""
    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        return  # tracing is off: do not even create a client
    try:
        from langsmith import run_trees

        client = run_trees.get_cached_client()
        client.flush()
    except Exception:
        pass
