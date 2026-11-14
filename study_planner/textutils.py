"""Tokenisation helpers shared by the embedder and the safety/grounding checks."""
from __future__ import annotations

import re
from typing import List, Set

STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers him his how i if in into is it its just me more most my no nor not of off on
    once only or other our out over own same she should so some such than that the their them then there
    these they this those through to too under until up very was we were what when where which while who
    whom why will with would you your yours""".split()
)

# Words that appear in *instructions* ("read", "summarize") rather than in course facts.
# They are ignored by the grounding check so that study-task wording is not penalised.
INSTRUCTION_WORDS = frozenset(
    """read review summarize summarise practice practise solve write explain compare list describe identify
    understand revise study complete attempt create make try use apply check test quiz note notes own words
    section sections chapter week weeks minute minutes task tasks step steps question questions answer answers
    exercise exercises flashcard flashcards focus finally first second next after before using based material
    materials course lecture given following above below key main core close book short summary gap gaps
    restate point sentence opening idea spend then once again compare against attempt work working through
    worked example examples solution solutions problem problems define definition definitions term terms
    recall mistakes mistake common topic topics concept concepts redo quickly quick""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER_RE = re.compile(r"\d")


def stem(word: str) -> str:
    """A tiny, dependency-free stemmer. Consistency matters more than linguistics."""
    if _NUMBER_RE.search(word):
        return word
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


INSTRUCTION_STEMS = frozenset(stem(w) for w in INSTRUCTION_WORDS)


def tokenize(text: str) -> List[str]:
    """Lower-cased, stop-word-free, stemmed tokens (numbers included)."""
    out: List[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        if len(raw) == 1 and not raw.isdigit():
            continue
        out.append(stem(raw))
    return out


def content_tokens(text: str, include_numbers: bool = False) -> Set[str]:
    """Distinct content-bearing word stems (used for grounding checks)."""
    result: Set[str] = set()
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS or raw in INSTRUCTION_WORDS:
            continue
        if _NUMBER_RE.search(raw):
            if include_numbers:
                result.add(raw)
            continue
        if len(raw) == 1:
            continue
        s = stem(raw)
        if s in INSTRUCTION_STEMS:
            continue
        result.add(s)
    return result


def normalize_key(text: str) -> str:
    return " ".join(sorted(set(tokenize(text))))
