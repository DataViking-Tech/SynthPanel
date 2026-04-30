"""Diff two saved panel results (sy-1b3n / GH-349)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synth_panel.convergence import (
    extract_category,
    identify_tracked_questions,
    jensen_shannon_divergence,
)
from synth_panel.stats import chi_squared_test

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class MetadataDiff:
    result_a_id: str
    result_b_id: str
    created_at_a: str
    created_at_b: str
    model_a: str
    model_b: str
    persona_count_a: int
    persona_count_b: int
    question_count_a: int
    question_count_b: int
    cost_a: str
    cost_b: str
    usage_a: dict[str, Any]
    usage_b: dict[str, Any]


@dataclass
class CategoricalQuestionDiff:
    question_text: str
    question_key: str
    distribution_a: dict[str, int]
    distribution_b: dict[str, int]
    jsd: float
    cramers_v_a: float | None
    cramers_v_b: float | None


@dataclass
class TextQuestionDiff:
    question_text: str
    question_key: str
    top_themes_a: list[str]
    top_themes_b: list[str]
    new_themes: list[str]
    dropped_themes: list[str]


@dataclass
class RunDiff:
    metadata: MetadataDiff
    categorical_questions: list[CategoricalQuestionDiff] = field(default_factory=list)
    text_questions: list[TextQuestionDiff] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_result(id_or_path: str) -> dict[str, Any]:
    """Load a panel result by ID or file path.

    Raises FileNotFoundError if neither path nor ID resolves.
    """
    p = Path(id_or_path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if "id" not in data:
            data["id"] = p.stem
        return data
    # Path-like strings (contain separator or extension) that don't exist as files
    # should raise FileNotFoundError rather than fall through to the ID lookup,
    # which would raise a misleading ValueError on path-traversal validation.
    if "/" in id_or_path or p.suffix:
        raise FileNotFoundError(f"Result file not found: {id_or_path!r}")
    from synth_panel.mcp.data import get_panel_result

    return get_panel_result(id_or_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "i",
        "my",
        "me",
        "we",
        "our",
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "not",
        "no",
        "so",
        "if",
        "when",
        "what",
        "how",
        "why",
        "which",
        "who",
        "just",
        "also",
        "more",
        "very",
        "much",
        "like",
        "really",
        "about",
        "there",
        "their",
        "they",
        "them",
        "then",
        "than",
    }
)


def _question_key(question: Any, index: int) -> str:
    if not isinstance(question, dict):
        return f"q{index}"
    for candidate in ("key", "id", "name"):
        val = question.get(candidate)
        if isinstance(val, str) and val:
            return val
    text = question.get("text")
    if isinstance(text, str) and text:
        return text.strip()[:40] or f"q{index}"
    return f"q{index}"


def _main_responses(panelist: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in panelist.get("responses", []) if isinstance(r, dict) and not r.get("follow_up")]


def _build_categorical_dist(
    panelists: list[dict[str, Any]],
    q_idx: int,
) -> dict[str, int]:
    dist: dict[str, int] = {}
    for p in panelists:
        responses = _main_responses(p)
        if q_idx < len(responses):
            cat = extract_category(responses[q_idx])
            if cat is not None:
                dist[cat] = dist.get(cat, 0) + 1
    return dist


def _extract_text_responses(
    panelists: list[dict[str, Any]],
    q_idx: int,
) -> list[str]:
    texts: list[str] = []
    for p in panelists:
        responses = _main_responses(p)
        if q_idx < len(responses):
            resp = responses[q_idx].get("response", "")
            if isinstance(resp, str) and resp.strip():
                texts.append(resp.strip())
    return texts


def _top_words(texts: list[str], n: int = 10) -> list[str]:
    words: list[str] = []
    for text in texts:
        for raw in text.lower().split():
            word = raw.strip(".,!?;:\"'()-[]")
            if len(word) >= 4 and word not in _STOP_WORDS:
                words.append(word)
    return [w for w, _ in Counter(words).most_common(n)]


def _cramers_v(dist: dict[str, int]) -> float | None:
    if not dist or sum(dist.values()) == 0:
        return None
    try:
        return chi_squared_test(dist).cramers_v
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core diff computation
# ---------------------------------------------------------------------------


def compute_diff(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
) -> RunDiff:
    """Compute a diff between two saved panel results."""
    metadata = MetadataDiff(
        result_a_id=result_a.get("id", ""),
        result_b_id=result_b.get("id", ""),
        created_at_a=result_a.get("created_at", ""),
        created_at_b=result_b.get("created_at", ""),
        model_a=result_a.get("model", ""),
        model_b=result_b.get("model", ""),
        persona_count_a=result_a.get("persona_count", 0),
        persona_count_b=result_b.get("persona_count", 0),
        question_count_a=result_a.get("question_count", 0),
        question_count_b=result_b.get("question_count", 0),
        cost_a=result_a.get("total_cost", ""),
        cost_b=result_b.get("total_cost", ""),
        usage_a=result_a.get("total_usage", {}),
        usage_b=result_b.get("total_usage", {}),
    )

    questions_a = result_a.get("questions") or []
    questions_b = result_b.get("questions") or []
    panelists_a = result_a.get("results") or []
    panelists_b = result_b.get("results") or []

    # Use longer question list as reference; fall back to question count
    questions = questions_a if len(questions_a) >= len(questions_b) else questions_b
    if not questions:
        # No question metadata saved — nothing to diff per-question
        return RunDiff(metadata=metadata)

    tracked = identify_tracked_questions(questions)
    tracked_indices = {idx for idx, _, _ in tracked}

    categorical: list[CategoricalQuestionDiff] = []
    text_qs: list[TextQuestionDiff] = []

    for i, q in enumerate(questions):
        q_text = q.get("text", f"Question {i + 1}") if isinstance(q, dict) else f"Question {i + 1}"
        q_key = _question_key(q, i)

        if i in tracked_indices:
            dist_a = _build_categorical_dist(panelists_a, i)
            dist_b = _build_categorical_dist(panelists_b, i)
            jsd = jensen_shannon_divergence(
                {k: float(v) for k, v in dist_a.items()},
                {k: float(v) for k, v in dist_b.items()},
            )
            categorical.append(
                CategoricalQuestionDiff(
                    question_text=q_text,
                    question_key=q_key,
                    distribution_a=dist_a,
                    distribution_b=dist_b,
                    jsd=jsd,
                    cramers_v_a=_cramers_v(dist_a),
                    cramers_v_b=_cramers_v(dist_b),
                )
            )
        else:
            texts_a = _extract_text_responses(panelists_a, i)
            texts_b = _extract_text_responses(panelists_b, i)
            top_a = _top_words(texts_a)
            top_b = _top_words(texts_b)
            set_a = set(top_a)
            set_b = set(top_b)
            text_qs.append(
                TextQuestionDiff(
                    question_text=q_text,
                    question_key=q_key,
                    top_themes_a=top_a,
                    top_themes_b=top_b,
                    new_themes=[w for w in top_b if w not in set_a][:5],
                    dropped_themes=[w for w in top_a if w not in set_b][:5],
                )
            )

    return RunDiff(
        metadata=metadata,
        categorical_questions=categorical,
        text_questions=text_qs,
    )
