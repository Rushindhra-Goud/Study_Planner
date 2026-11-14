"""Deterministic scheduling: spread topics over weeks and divide each week's time."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass
class WeekSlot:
    week: int
    kind: str  # study | revision
    topics: List[str] = field(default_factory=list)


def partition(weights: Sequence[float], k: int) -> List[List[int]]:
    """Split indices 0..n-1 into k contiguous, non-empty groups with roughly equal total weight."""
    n = len(weights)
    if k < 1 or n < k:
        raise ValueError("need 1 <= k <= number of items")
    groups: List[List[int]] = []
    i = 0
    remaining = float(sum(weights))
    for g in range(k):
        groups_left = k - g
        if groups_left == 1:
            groups.append(list(range(i, n)))
            break
        target = remaining / groups_left
        current: List[int] = []
        total = 0.0
        while i < n and (n - i) > (groups_left - 1):
            w = float(weights[i])
            if current and total + w / 2.0 > target:
                break
            current.append(i)
            total += w
            i += 1
        groups.append(current)
        remaining -= total
    return groups


def build_schedule(topics: Sequence[str], weights: Sequence[float], weeks: int,
                   weak: Sequence[str] = ()) -> Tuple[List[WeekSlot], List[str]]:
    """Return (week slots, warnings). Never creates more study weeks than there are topics."""
    if not topics:
        return [], ["No topics available to schedule."]
    warnings: List[str] = []
    n = len(topics)
    study_weeks = min(weeks, n)
    revision_weeks = weeks - study_weeks
    if revision_weeks > 0:
        revision_weeks = min(revision_weeks, max(1, study_weeks // 2))
    total_weeks = study_weeks + revision_weeks
    if total_weeks < weeks:
        warnings.append(
            f"The course material only has {n} topics, so the plan was shortened from {weeks} to {total_weeks} weeks."
        )

    slots: List[WeekSlot] = []
    for number, group in enumerate(partition(list(weights), study_weeks), start=1):
        slots.append(WeekSlot(week=number, kind="study", topics=[topics[i] for i in group]))

    if revision_weeks:
        weak_set = {w for w in weak}
        ranked = [t for t in topics if t in weak_set] + sorted(
            (t for t in topics if t not in weak_set), key=lambda t: -weights[list(topics).index(t)]
        )
        per_week = min(3, len(ranked))
        for r in range(revision_weeks):
            picked = [ranked[(r * per_week + j) % len(ranked)] for j in range(per_week)]
            unique = list(dict.fromkeys(picked))
            slots.append(WeekSlot(week=study_weeks + r + 1, kind="revision", topics=unique))
    return slots, warnings


def split_minutes(total_minutes: float, kind: str) -> Tuple[int, int, int]:
    """(study, practice, review) minutes for one week."""
    shares = (0.65, 0.25, 0.10) if kind == "study" else (0.40, 0.50, 0.10)
    study = int(round(total_minutes * shares[0]))
    practice = int(round(total_minutes * shares[1]))
    review = max(0, int(round(total_minutes)) - study - practice)
    return study, practice, review


def scale_minutes(raw: Sequence[int], budget: int) -> List[int]:
    """Scale task durations so they add up to `budget` (largest-remainder rounding)."""
    if not raw:
        return []
    budget = max(budget, len(raw))
    weights = [max(1, int(r)) for r in raw]
    total = sum(weights)
    exact = [budget * w / total for w in weights]
    floors = [int(x) for x in exact]
    remainder = budget - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors
