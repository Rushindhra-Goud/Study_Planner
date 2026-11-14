"""Load course files (.md, .txt, .pdf), sanitise them, split into topic-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import Chunk
from .safety import sanitize_document

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
MIN_CHUNK_CHARS = 40

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PLAIN_HEADING_PATTERNS = [
    re.compile(r"^(chapter|unit|module|week|lecture|section|topic)\s+\d+\b.*$", re.I),
    re.compile(r"^\d+(\.\d+){0,2}[.)]?\s+[A-Z][^.!?]{2,70}$"),
]
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`*])")


@dataclass
class Unit:
    topic: str
    heading_path: str
    text: str


@dataclass
class IngestResult:
    chunks: List[Chunk] = field(default_factory=list)
    documents: List[Dict[str, object]] = field(default_factory=list)
    removed_lines: List[Tuple[str, str]] = field(default_factory=list)  # (source, line)
    skipped: List[str] = field(default_factory=list)


def pretty_title(stem: str) -> str:
    name = re.sub(r"^\d+[_\-\s]*", "", stem)
    name = name.replace("_", " ").replace("-", " ").strip()
    return name.title() if name else stem


def _clean_heading(title: str) -> str:
    return re.sub(r"[*`#_]+", "", title).strip()


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if any(p.match(s) for p in _PLAIN_HEADING_PATTERNS):
        return True
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 4 and s.upper() == s and len(s.split()) <= 8 and not s.endswith(".")


# ---------------------------------------------------------------- loading
def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pypdf is required to read PDF files: pip install pypdf") from exc
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # a single bad page should not stop the ingest
                pages.append("")
        return "\n\n".join(pages)
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- parsing
def parse_markdown(text: str, fallback_title: str) -> Tuple[str, List[Unit]]:
    """Return (document title, units). A unit is one paragraph with its topic and heading path."""
    units: List[Unit] = []
    path: List[str] = []
    topic = ""
    doc_title = ""
    buf: List[str] = []
    in_fence = False

    def flush() -> None:
        para = "\n".join(buf).strip()
        buf.clear()
        if para:
            units.append(Unit(topic or fallback_title, " > ".join(path) or fallback_title, para))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        if in_fence:
            buf.append(line)
            continue
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = _clean_heading(m.group(2))
            if not title:
                continue
            path = path[: level - 1] + [title]
            if level == 1 and not doc_title:
                doc_title = title
            if level <= 2 or not topic:
                topic = title
            continue
        if not stripped:
            flush()
            continue
        buf.append(line)
    flush()
    return doc_title or fallback_title, units


def parse_plain(text: str, fallback_title: str) -> Tuple[str, List[Unit]]:
    units: List[Unit] = []
    topic = ""
    buf: List[str] = []

    def flush() -> None:
        para = " ".join(s.strip() for s in buf).strip()
        buf.clear()
        if para:
            units.append(Unit(topic or fallback_title, topic or fallback_title, para))

    for line in text.splitlines():
        if _looks_like_heading(line):
            flush()
            topic = _clean_heading(line)
            continue
        if not line.strip():
            flush()
            continue
        buf.append(line)
    flush()
    return fallback_title, units


# ---------------------------------------------------------------- chunking
def _split_long(text: str, max_chars: int) -> List[str]:
    sentences = _SENTENCE_SPLIT.split(text)
    pieces: List[str] = []
    cur = ""
    for sentence in sentences:
        while len(sentence) > max_chars:  # extremely long sentence: cut on spaces
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if cur and len(cur) + 1 + len(sentence) > max_chars:
            pieces.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p]


def pack_units(units: List[Unit], max_chars: int) -> List[Tuple[str, str, str]]:
    """Greedily pack paragraphs of the same topic into chunks of <= max_chars."""
    out: List[Tuple[str, str, str]] = []
    cur_topic: Optional[str] = None
    cur_path = ""
    cur_parts: List[str] = []
    cur_len = 0

    def emit() -> None:
        nonlocal cur_parts, cur_len
        if cur_parts:
            out.append((cur_topic or "", cur_path, "\n\n".join(cur_parts)))
        cur_parts, cur_len = [], 0

    for unit in units:
        pieces = [unit.text] if len(unit.text) <= max_chars else _split_long(unit.text, max_chars)
        for piece in pieces:
            if unit.topic != cur_topic:
                emit()
                cur_topic = unit.topic
            if cur_parts and cur_len + len(piece) + 2 > max_chars:
                emit()
            if not cur_parts:
                cur_path = unit.heading_path
            cur_parts.append(piece)
            cur_len += len(piece) + 2
    emit()
    return out


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def chunk_document(path: Path, raw_text: str, max_chars: int) -> Tuple[List[Chunk], str, List[str]]:
    clean, removed = sanitize_document(raw_text)
    fallback = pretty_title(path.stem)
    if path.suffix.lower() in (".md", ".markdown"):
        title, units = parse_markdown(clean, fallback)
    else:
        title, units = parse_plain(clean, fallback)
    stem = _safe_stem(path)
    chunks: List[Chunk] = []
    for topic, heading_path, text in pack_units(units, max_chars):
        if len(text.strip()) < MIN_CHUNK_CHARS:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{stem}#{len(chunks) + 1}",
                source=path.name,
                doc_title=title,
                topic=topic,
                heading_path=heading_path,
                text=text.strip(),
            )
        )
    return chunks, title, removed


def _disambiguate_topics(chunks: List[Chunk]) -> None:
    """If two documents use the same section name, append the document title."""
    owners: Dict[str, set] = {}
    for c in chunks:
        owners.setdefault(c.topic, set()).add(c.source)
    for c in chunks:
        if len(owners[c.topic]) > 1:
            c.topic = f"{c.topic} ({c.doc_title})"


def ingest_directory(data_dir: Path, chunk_chars: int = 900) -> IngestResult:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Course material directory not found: {data_dir}")
    result = IngestResult()
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in files:
        try:
            raw = load_text(path)
        except Exception as exc:
            result.skipped.append(f"{path.name}: {exc}")
            continue
        if not raw.strip():
            result.skipped.append(f"{path.name}: no extractable text")
            continue
        chunks, title, removed = chunk_document(path, raw, chunk_chars)
        if not chunks:
            result.skipped.append(f"{path.name}: no usable content")
            continue
        result.chunks.extend(chunks)
        result.documents.append({"source": path.name, "title": title, "chunks": len(chunks)})
        result.removed_lines.extend((path.name, line) for line in removed)
    _disambiguate_topics(result.chunks)
    return result
