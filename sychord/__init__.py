"""Public package entry-point for the Syntakt SY CHORD translator."""
from __future__ import annotations

from .translator import (
    ANALYSIS_JSON_SCHEMA,
    AnalysisResult,
    MatchCandidate,
    Session,
    advise_keyboard_scale,
    analyze_text,
    format_analysis_fr,
    format_for_syntakt,
    get_doc,
    get_session,
    get_version,
    has_pythonista_ui,
    normalize_input,
    present_ui,
)

__all__ = [
    "Session",
    "AnalysisResult",
    "MatchCandidate",
    "ANALYSIS_JSON_SCHEMA",
    "analyze_text",
    "advise_keyboard_scale",
    "format_analysis_fr",
    "format_for_syntakt",
    "get_doc",
    "get_session",
    "get_version",
    "has_pythonista_ui",
    "normalize_input",
    "present_ui",
]
