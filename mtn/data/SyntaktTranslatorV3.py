"""Façade haut niveau pour l'analyse des accords Syntakt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from syntakt_core import (
    ANALYSIS_JSON_SCHEMA,
    ENGINE_VERSION,
    NormalizedInput,
    load_library,
    normalize_input,
    rank_presets,
)

__all__ = ["Session", "format_analysis_fr", "ANALYSIS_JSON_SCHEMA"]


@dataclass
class AnalysisResult:
    input: NormalizedInput
    options: Dict[str, Any]
    best: Optional[Dict[str, Any]]
    alternatives: List[Dict[str, Any]]
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "input": self.input.as_dict(),
            "options": dict(self.options),
            "best": self.best,
            "alternatives": self.alternatives,
            "copy_lines": [cand["copy_line"] for cand in self.alternatives],
            "engine_version": ENGINE_VERSION,
            "schema": ANALYSIS_JSON_SCHEMA,
            "error": self.error,
        }
        payload["chord_pcs"] = sorted(self.input.pcs_set)
        return payload


class Session:
    """Point d'entrée principal pour les analyses SY CHORD."""

    def __init__(
        self,
        *,
        strict: bool = False,
        pref_bass_root: bool = True,
        pref_root_ident: bool = False,
        anchor_octave: int = 3,
        topk: int = 12,
    ) -> None:
        self._defaults = {
            "strict": strict,
            "pref_bass_root": pref_bass_root,
            "pref_root_ident": pref_root_ident,
            "anchor_octave": anchor_octave,
            "topk": topk,
        }
        self._library = load_library()

    @property
    def library(self):
        return self._library

    def analyze(self, user_input: str, **overrides: Any) -> Dict[str, Any]:
        options = dict(self._defaults)
        options.update(overrides)
        try:
            normalized = normalize_input(user_input)
        except ValueError as exc:
            result = AnalysisResult(
                input=NormalizedInput(
                    raw=user_input,
                    kind="error",
                    root=None,
                    quality=None,
                    notes=tuple(),
                    pcs_set=frozenset(),
                    rel=tuple(),
                    inversion=None,
                    bass=None,
                    expected_quality=None,
                    expected_intervals=tuple(),
                ),
                options=options,
                best=None,
                alternatives=[],
                error=str(exc),
            )
            return result.to_dict()

        scored = rank_presets(
            normalized,
            strict=options["strict"],
            pref_bass_root=options["pref_bass_root"],
            pref_root_ident=options["pref_root_ident"],
        )
        topk = max(1, int(options.get("topk", 12)))
        candidates: List[Dict[str, Any]] = []
        for preset, stars, metrics, metrics_raw in scored[:topk]:
            payload = preset.as_dict(
                options["anchor_octave"],
                stars=stars,
                metrics=metrics,
                metrics_raw=metrics_raw,
            )
            candidates.append(payload)
        best = candidates[0] if candidates else None
        result = AnalysisResult(
            input=normalized,
            options=options,
            best=best,
            alternatives=candidates,
            error=None,
        )
        return result.to_dict()


def format_analysis_fr(result: Dict[str, Any]) -> str:
    if result.get("error"):
        return f"❌ Analyse impossible : {result['error']}"
    best = result.get("best")
    if not best:
        return "⚠️ Aucun preset approprié n'a été trouvé."
    lines = [
        "Résultat :",
        f"- Réglage recommandé : {best['copy_line']}",
        "- Notes jouées : " + ", ".join(best["notes_oct"]),
    ]
    if best["metrics"].get("explanation"):
        lines.append(f"- Qualité du match : {best['metrics']['explanation']} ({best['stars']}★)")
    alternatives = result.get("alternatives", [])[1:]
    if alternatives:
        lines.append("- Alternatives top-k :")
        for alt in alternatives:
            lines.append(f"  • {alt['stars']}★ {alt['copy_line']}")
    return "\n".join(lines)
