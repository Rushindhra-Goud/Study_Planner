"""Safety rules that keep the planner from hallucinating.

Rules (each is enforced in code, counted in the SafetyReport and shown in the plan):

 R1  Prompt-injection lines found in course files are removed at ingestion.
 R2  A topic with no sufficiently relevant course material is REFUSED, not invented.
 R3  Every task / question must cite at least one retrieved chunk; unknown citations are discarded.
 R4  Every question and solution must be lexically grounded in the cited context
     (share of content words that appear in the retrieved material).
 R5  URLs that do not appear in the course material are stripped from the output.
 R6  Simple arithmetic written as "a op b = c" is re-computed; wrong equations are dropped.
 R7  Numbers in a solution that cannot be traced to the material, the question or a verified
     equation are flagged `needs_review` for a human (HITL).
 R8  Malformed / too-short generator output is dropped.
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .models import ContextChunk, Question, Task
from .textutils import content_tokens

# --------------------------------------------------------------------------- R1
_INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,40}\b(instruction|prompt|rule)s?\b", re.I),
    re.compile(r"\byou are now (a|an|in)\b", re.I),
    re.compile(r"\b(reveal|print|show|output|leak)\b.{0,30}\b(system prompt|api key|secret|password)s?\b", re.I),
    re.compile(r"</?\s*(system|assistant|user)\s*>", re.I),
    re.compile(r"\bnew instructions\s*:", re.I),
    re.compile(r"\bdo not (follow|obey) (the )?(above|previous)", re.I),
]


def sanitize_document(text: str) -> Tuple[str, List[str]]:
    """Remove lines that look like prompt-injection attempts. Returns (clean_text, removed_lines)."""
    kept: List[str] = []
    removed: List[str] = []
    for line in text.splitlines():
        if any(p.search(line) for p in _INJECTION_PATTERNS):
            removed.append(line.strip()[:200])
        else:
            kept.append(line)
    return "\n".join(kept), removed


# --------------------------------------------------------------------------- R5
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s)>\]]+", re.I)


def strip_untrusted_urls(text: str, allowed_text: str) -> Tuple[str, int]:
    count = 0
    allowed_lower = allowed_text.lower()

    def repl(match: "re.Match[str]") -> str:
        nonlocal count
        url = match.group(0).rstrip(".,;:!?")
        if url.lower() in allowed_lower:
            return match.group(0)
        count += 1
        return "[link removed]"

    return _URL_RE.sub(repl, text), count


# --------------------------------------------------------------------------- R4
@dataclass
class Support:
    score: float
    unsupported: List[str]


def grounding(text: str, allowed_text: str) -> Support:
    """Share of content words of `text` that also occur in `allowed_text` (numbers excluded)."""
    words = content_tokens(text)
    if not words:
        return Support(1.0, [])
    allowed = content_tokens(allowed_text)
    prefixes = {a[:5] for a in allowed if len(a) >= 5}
    unsupported = sorted(w for w in words if w not in allowed and not (len(w) >= 5 and w[:5] in prefixes))
    return Support(1.0 - len(unsupported) / len(words), unsupported)


# --------------------------------------------------------------------------- R6
_ALLOWED_CHARS = set("0123456789.+-*/×÷() \t")
_OPS = "+-*/×÷"
_PLAIN_NUMBER = re.compile(r"-?\d+(\.\d+)?")


def _eval_expr(expr: str) -> Optional[float]:
    expr = expr.replace("×", "*").replace("÷", "/").strip()
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = ev(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            return a / b
        raise ValueError("unsupported expression")

    try:
        return ev(tree)
    except (ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def _has_operator(expr: str) -> bool:
    return re.search(r"(?<=[\d)])\s*[+\-*/×÷]", expr) is not None


def _tolerance(left: str, right: str, a: float, b: float) -> float:
    """Rounding-aware tolerance: '10 / 3 = 3.33' is fine, '2 + 2 = 5' is not."""
    for side in (left, right):
        if _PLAIN_NUMBER.fullmatch(side):
            decimals = len(side.split(".")[1]) if "." in side else 0
            return 0.5 * 10 ** (-decimals) + 1e-9
    return 1e-6 * max(1.0, abs(a), abs(b))


@dataclass
class ArithmeticResult:
    errors: List[str] = field(default_factory=list)
    verified_numbers: Set[str] = field(default_factory=set)
    checked: int = 0


def norm_number(token: str) -> str:
    try:
        return str(float(token))
    except ValueError:
        return token


def check_arithmetic(text: str) -> ArithmeticResult:
    """Re-compute equations such as '40 / (40 + 10) = 0.8' that appear in `text`."""
    result = ArithmeticResult()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    n = len(text)
    for m in re.finditer("=", text):
        i = m.start()
        j = i
        while j > 0 and text[j - 1] in _ALLOWED_CHARS:
            j -= 1
        left_run = text[j:i]
        k = i + 1
        while k < n and text[k] in _ALLOWED_CHARS:
            k += 1
        right_run = text[i + 1 : k]

        left = left_run.strip()
        right = right_run.strip()
        if right.endswith("."):
            right = right[:-1].rstrip()
        if not left or not right:
            continue
        if left[0] in _OPS or left[-1] in _OPS or right[-1] in _OPS:
            continue
        if right[0] in "*/×÷+" or not re.search(r"\d", left) or not re.search(r"\d", right):
            continue
        if not (_has_operator(left) or _has_operator(right)):
            continue

        # Boundary checks: skip things like "x^2 + 3 = 7" or "2x + 1 = 7".
        start = j + (len(left_run) - len(left_run.lstrip()))
        if start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_^)"):
            continue
        k_back = start - 1
        while k_back >= 0 and text[k_back] == " ":
            k_back -= 1
        if k_back >= 0 and text[k_back] in "^":
            continue
        end = i + 1 + len(right_run.rstrip())
        if end < n and (text[end].isalpha() or text[end] in "^%_"):
            continue

        a = _eval_expr(left)
        b = _eval_expr(right)
        if a is None or b is None:
            continue
        result.checked += 1
        if abs(a - b) <= _tolerance(left, right, a, b):
            for num in re.findall(r"\d+(?:\.\d+)?", left + " " + right):
                result.verified_numbers.add(norm_number(num))
        else:
            result.errors.append(f"{left} = {right} (computed {a:.4g} vs {b:.4g})")
    return result


def find_arithmetic_errors(text: str) -> List[str]:
    return check_arithmetic(text).errors


# --------------------------------------------------------------------------- R7
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text: str) -> Set[str]:
    return {norm_number(n) for n in _NUM_RE.findall(text)}


def derived_numbers(solution: str, allowed_text: str, verified: Iterable[str] = ()) -> List[str]:
    """Numbers in a solution that cannot be traced to the sources. Small integers (<=10) are ignored."""
    allowed = numbers_in(allowed_text) | set(verified)
    out = []
    for token in sorted(numbers_in(solution), key=lambda s: float(s)):
        if token in allowed:
            continue
        value = float(token)
        if value.is_integer() and value <= 10:
            continue
        out.append(token)
    return out


# --------------------------------------------------------------------------- R3
def normalise_sids(raw: Any) -> List[str]:
    """Accept ['S1'], '[S1]', 'S1, S2', [1, 2] ... and return ['S1', 'S2']."""
    if raw is None:
        return []
    items: List[Any]
    if isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = [raw]
    out: List[str] = []
    for item in items:
        for part in re.split(r"[,\s]+", str(item)):
            part = part.strip("[]() ").upper()
            if not part:
                continue
            if part.isdigit():
                part = "S" + part
            if re.fullmatch(r"S\d+", part) and part not in out:
                out.append(part)
    return out


def validate_citations(raw: Any, ctx_map: Dict[str, ContextChunk]) -> List[str]:
    """Return the chunk ids behind the *valid* citations; unknown ids are discarded."""
    chunk_ids: List[str] = []
    for sid in normalise_sids(raw):
        ctx = ctx_map.get(sid)
        if ctx is not None and ctx.chunk.chunk_id not in chunk_ids:
            chunk_ids.append(ctx.chunk.chunk_id)
    return chunk_ids


# --------------------------------------------------------------------------- report
@dataclass
class SafetyReport:
    injection_lines_removed: int = 0
    topics_refused: List[str] = field(default_factory=list)
    items_checked: int = 0
    items_dropped: Dict[str, int] = field(default_factory=dict)
    items_flagged: int = 0
    urls_stripped: int = 0
    revisions_used: int = 0
    llm_failures: int = 0
    fallbacks_used: int = 0

    def drop(self, reason: str) -> None:
        self.items_dropped[reason] = self.items_dropped.get(reason, 0) + 1

    @property
    def total_dropped(self) -> int:
        return sum(self.items_dropped.values())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["total_dropped"] = self.total_dropped
        return data


# --------------------------------------------------------------------------- policy
def _context_text(ctx: Sequence[ContextChunk]) -> str:
    return "\n".join(f"{c.chunk.heading_path}\n{c.chunk.text}" for c in ctx)


def _cited_context_text(ctx: Sequence[ContextChunk], chunk_ids: Sequence[str]) -> str:
    cited = [c for c in ctx if c.chunk.chunk_id in set(chunk_ids)]
    return _context_text(cited or ctx)


class SafetyPolicy:
    def __init__(self, min_grounding: float = 0.6, min_question_grounding: float = 0.5, min_task_grounding: float = 0.3):
        self.min_grounding = min_grounding
        self.min_question_grounding = min_question_grounding
        self.min_task_grounding = min_task_grounding
        self.report = SafetyReport()

    def reset(self) -> None:
        self.report = SafetyReport()

    # -- tasks -----------------------------------------------------------
    def check_task(self, raw: Dict[str, Any], ctx: Sequence[ContextChunk]) -> Tuple[Optional[Task], str]:
        self.report.items_checked += 1
        ctx_map = {c.sid: c for c in ctx}
        title = str(raw.get("title", "")).strip()
        description = str(raw.get("description", "")).strip()
        if len(title) < 3 or len(description) < 10:
            return None, "malformed_task"
        chunk_ids = validate_citations(raw.get("citations"), ctx_map)
        if not chunk_ids:
            return None, "task_without_valid_citation"
        all_text = _context_text(ctx)
        description, n1 = strip_untrusted_urls(description, all_text)
        title, n2 = strip_untrusted_urls(title, all_text)
        self.report.urls_stripped += n1 + n2
        support = grounding(title + ". " + description, _cited_context_text(ctx, chunk_ids) + "\n" + all_text)
        if support.score < self.min_task_grounding:
            return None, "task_ungrounded"
        try:
            minutes = int(round(float(raw.get("minutes", 20))))
        except (TypeError, ValueError):
            minutes = 20
        minutes = max(5, min(180, minutes))
        return Task(title=title, description=description, minutes=minutes, citations=chunk_ids, support=round(support.score, 3)), ""

    # -- questions -------------------------------------------------------
    def check_question(self, raw: Dict[str, Any], ctx: Sequence[ContextChunk]) -> Tuple[Optional[Question], str]:
        self.report.items_checked += 1
        ctx_map = {c.sid: c for c in ctx}
        question = str(raw.get("question", "")).strip()
        solution = str(raw.get("solution", "")).strip()
        if len(question) < 10 or len(solution) < 15:
            return None, "malformed_question"
        kind = str(raw.get("source_kind", "llm"))
        difficulty = str(raw.get("difficulty", "medium")).lower()
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"

        chunk_ids = validate_citations(raw.get("citations"), ctx_map)
        if not chunk_ids:
            return None, "question_without_valid_citation"

        all_text = _context_text(ctx)
        question, n1 = strip_untrusted_urls(question, all_text)
        solution, n2 = strip_untrusted_urls(solution, all_text)
        self.report.urls_stripped += n1 + n2

        cited_text = _cited_context_text(ctx, chunk_ids)
        q_support = grounding(question, cited_text + "\n" + all_text)
        s_support = grounding(solution, cited_text + "\n" + all_text + "\n" + question)
        if q_support.score < self.min_question_grounding:
            return None, "question_ungrounded"
        if s_support.score < self.min_grounding:
            return None, "solution_ungrounded"

        arithmetic = check_arithmetic(solution + "\n" + question)
        if arithmetic.errors:
            return None, "arithmetic_error"

        needs_review = False
        reason = ""
        if kind != "worked_example":
            extra = derived_numbers(solution, cited_text + "\n" + all_text + "\n" + question, arithmetic.verified_numbers)
            if extra:
                needs_review = True
                reason = "numbers not found in the material or in a verified equation: " + ", ".join(extra[:6])
                self.report.items_flagged += 1

        return (
            Question(
                question=question,
                solution=solution,
                difficulty=difficulty,
                citations=chunk_ids,
                source_kind=kind,
                support=round(min(q_support.score, s_support.score), 3),
                needs_review=needs_review,
                review_reason=reason,
            ),
            "",
        )
