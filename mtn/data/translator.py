"""High-level helpers for the Syntakt SY CHORD translator."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .core import (
    ANALYSIS_JSON_SCHEMA,
    AnalysisResult,
    MatchCandidate,
    Session,
    aggregated_chord_pcs_from_results,
    format_analysis_fr,
    format_for_syntakt,
    get_doc,
    get_version,
    normalize_input,
    recommend_kb_scale,
)

__all__ = [
    "Session",
    "AnalysisResult",
    "MatchCandidate",
    "ANALYSIS_JSON_SCHEMA",
    "format_analysis_fr",
    "format_for_syntakt",
    "normalize_input",
    "get_doc",
    "get_version",
    "analyze_text",
    "get_session",
    "advise_keyboard_scale",
]


_session: Optional[Session] = None


def get_session() -> Session:
    """Return a shared :class:`Session` instance."""

    global _session
    if _session is None:
        _session = Session()
    return _session


def analyze_text(input_text: str, **options: Any) -> Dict[str, Any]:
    """Shortcut for ``get_session().analyze``."""

    return get_session().analyze(input_text, **options)


def advise_keyboard_scale(symbols: Sequence[str], policy: str = "safe") -> List[Dict[str, Any]]:
    """Return the top keyboard scale recommendations for a chord progression."""

    analyses = [get_session().analyze(sym) for sym in symbols]
    pcs = aggregated_chord_pcs_from_results(analyses)
    top = recommend_kb_scale(pcs, policy=policy)
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return [
        {"kb_scale": scale, "root": note_names[root_pc], "score": round(score, 3)}
        for (scale, root_pc, score) in top
    ]


