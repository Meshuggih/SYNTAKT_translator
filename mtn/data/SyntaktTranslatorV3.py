"""Façade haut niveau pour le moteur SY CHORD."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Imports explicites depuis le moteur plat : aucun import relatif requis.
from syntakt_core import (
    ANALYSIS_JSON_SCHEMA,
    ENGINE_VERSION,
    NormalizedInput,
    Preset,
    QUALITY_INTERVALS,
    load_library,
    midi_from_note_oct,
    normalize_input,
    rank_presets,
)

__all__ = ["voicing_for", "analyze", "Session", "format_analysis_fr"]

_NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _interval_sequence(preset: Preset) -> List[int]:
    """Retourne la séquence d'intervalles tenant compte de l'inversion."""
    reference = QUALITY_INTERVALS.get(preset.sy_type)
    if reference:
        seq = list(reference)
        inv = (preset.inversion or 0) % len(seq) if seq else 0
        if inv:
            seq = seq[inv:] + [val + 12 for val in seq[:inv]]
            seq[0] -= 12
        return seq
    return list(preset.intervals)


def _voicing_midis(preset: Preset, anchor_octave: int) -> List[int]:
    intervals = _interval_sequence(preset)
    if not intervals:
        return []
    root_midi = midi_from_note_oct(preset.root, anchor_octave)
    midis = [root_midi + offset for offset in intervals]
    midis.sort()

    floor = midi_from_note_oct("C", anchor_octave)
    ceiling = midi_from_note_oct("C", anchor_octave + 2) - 1

    changed = True
    while changed:
        changed = False
        max_val = max(midis)
        min_val = min(midis)
        if max_val > ceiling:
            midis[midis.index(max_val)] -= 12
            changed = True
        if min_val < floor:
            midis[midis.index(min_val)] += 12
            changed = True
        if changed:
            midis.sort()

    for idx in range(1, len(midis)):
        while midis[idx] <= midis[idx - 1]:
            midis[idx] += 12
    midis.sort()

    while midis and midis[-1] - midis[0] > 12:
        midis[midis.index(min(midis))] += 12
        midis.sort()
        while True:
            adjust = False
            max_val = max(midis)
            min_val = min(midis)
            if max_val > ceiling:
                midis[midis.index(max_val)] -= 12
                adjust = True
            if min_val < floor:
                midis[midis.index(min_val)] += 12
                adjust = True
            if not adjust:
                break
            midis.sort()
        for idx in range(1, len(midis)):
            while midis[idx] <= midis[idx - 1]:
                midis[idx] += 12
        midis.sort()

    return midis


def _prefer_flats(root: str, prefer_flats: Optional[bool]) -> bool:
    if prefer_flats is not None:
        return prefer_flats
    return "b" in root and "#" not in root


def _note_label(midi: int, prefer_flats: bool) -> str:
    octave = midi // 12 - 1
    names = _NOTE_NAMES_FLAT if prefer_flats else _NOTE_NAMES_SHARP
    return f"{names[midi % 12]}{octave}"


def _strip_octave(label: str) -> str:
    for idx, char in enumerate(label):
        if char.isdigit() or char in {"-"}:
            return label[:idx]
    return label


def _voicing_labels(
    preset: Preset,
    anchor_octave: int,
    prefer_flats: Optional[bool],
) -> Tuple[List[str], List[int]]:
    midis = _voicing_midis(preset, anchor_octave)
    prefer = _prefer_flats(preset.root, prefer_flats)
    labels = [_note_label(midi, prefer) for midi in midis]
    return labels, midis


def voicing_for(preset: Preset, anchor_octave: int = 3, prefer_flats: bool | None = None) -> List[str]:
    """Calcule un voicing grave→aigu avec octaves pour un preset donné."""
    labels, _ = _voicing_labels(preset, anchor_octave, prefer_flats)
    return labels


def _candidate_payload(
    preset: Preset,
    *,
    stars: int,
    metrics: Dict[str, Any],
    metrics_raw: Sequence[int],
    anchor_octave: int,
    prefer_flats: Optional[bool],
) -> Dict[str, Any]:
    labels, midis = _voicing_labels(preset, anchor_octave, prefer_flats)
    notes = [_strip_octave(label) for label in labels]
    copy_line = f"Root={preset.root}  Preset={preset.sy_type}  BAL={preset.bal}"
    payload: Dict[str, Any] = {
        "root": preset.root,
        "preset": preset.sy_type,
        "preset_name": preset.name,
        "bal": preset.bal,
        "bal_pattern": preset.motif,
        "labels": list(preset.labels),
        "inversion": preset.inversion,
        "voicing": " ".join(labels),
        "voicing_list": list(labels),
        "notes": notes,
        "notes_oct": list(labels),
        "stars": int(stars),
        "metrics": dict(metrics),
        "metrics_raw": list(metrics_raw),
        "bal_preference_rank": metrics.get("bal_preference_rank", 9),
        "voicing_span": midis[-1] - midis[0] if len(midis) > 1 else 0,
        "copy_line": copy_line,
        "sy_root": preset.root,
        "sy_type": preset.sy_type,
        "bal_motif": preset.motif,
        "pcs": list(preset.pcs),
        "pitch_classes": list(preset.pcs),
    }
    payload["_pitch_set"] = frozenset(midi % 12 for midi in midis)
    payload["_triad_second_inversion"] = preset.nvoices == 3 and preset.inversion == 2
    return payload


def _apply_bal_tiebreak(candidates: List[Dict[str, Any]]) -> None:
    for idx in range(len(candidates) - 1):
        current = candidates[idx]
        nxt = candidates[idx + 1]
        if (
            current["stars"] == nxt["stars"]
            and current["_triad_second_inversion"]
            and nxt["_triad_second_inversion"]
            and current["_pitch_set"] == nxt["_pitch_set"]
            and current["bal"] != nxt["bal"]
            and nxt["bal"] == 106
            and current["bal"] != 106
        ):
            candidates[idx], candidates[idx + 1] = nxt, current


def _sanitize_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for entry in candidates:
        entry = dict(entry)
        entry.pop("_pitch_set", None)
        entry.pop("_triad_second_inversion", None)
        cleaned.append(entry)
    return cleaned


def analyze(
    user_input: str,
    *,
    strict: bool = False,
    pref_bass_root: bool = True,
    pref_root_ident: bool = False,
    anchor_octave: int = 3,
    topk: int = 12,
    prefer_flats: bool | None = None,
) -> Dict[str, Any]:
    """Pipeline haut-niveau : normalisation → ranking → voicings formatés."""
    options = {
        "strict": strict,
        "pref_bass_root": pref_bass_root,
        "pref_root_ident": pref_root_ident,
        "anchor_octave": anchor_octave,
        "topk": topk,
        "prefer_flats": prefer_flats,
    }

    try:
        normalized = normalize_input(user_input)
    except ValueError as exc:
        return {
            "input": {
                "raw": user_input,
                "kind": "error",
            },
            "options": options,
            "best": None,
            "alternatives": [],
            "candidates": [],
            "copy_line": None,
            "copy_lines": [],
            "stars": 0,
            "engine_version": ENGINE_VERSION,
            "schema": ANALYSIS_JSON_SCHEMA,
            "error": str(exc),
            "chord_pcs": [],
        }

    scored = rank_presets(
        normalized,
        strict=strict,
        pref_bass_root=pref_bass_root,
        pref_root_ident=pref_root_ident,
    )

    max_candidates = max(1, int(topk))
    raw_candidates: List[Dict[str, Any]] = []
    for preset, stars_value, metrics, metrics_raw in scored[:max_candidates]:
        raw_candidates.append(
            _candidate_payload(
                preset,
                stars=stars_value,
                metrics=metrics,
                metrics_raw=metrics_raw,
                anchor_octave=anchor_octave,
                prefer_flats=prefer_flats,
            )
        )

    if raw_candidates:
        _apply_bal_tiebreak(raw_candidates)

    candidates = _sanitize_candidates(raw_candidates)
    best = candidates[0] if candidates else None
    alternatives = candidates[1:] if len(candidates) > 1 else []

    result = {
        "input": normalized.as_dict(),
        "options": options,
        "best": best,
        "alternatives": alternatives,
        "candidates": candidates,
        "copy_line": best["copy_line"] if best else None,
        "copy_lines": [cand["copy_line"] for cand in candidates],
        "stars": best["stars"] if best else 0,
        "engine_version": ENGINE_VERSION,
        "schema": ANALYSIS_JSON_SCHEMA,
        "error": None,
        "chord_pcs": sorted(normalized.pcs_set),
    }
    return result


class Session:
    """Session pratique encapsulant le chargement de la librairie."""

    def __init__(
        self,
        *,
        strict: bool = False,
        pref_bass_root: bool = True,
        pref_root_ident: bool = False,
        anchor_octave: int = 3,
        topk: int = 12,
        csv_path: str | None = None,
    ) -> None:
        self._defaults = {
            "strict": strict,
            "pref_bass_root": pref_bass_root,
            "pref_root_ident": pref_root_ident,
            "anchor_octave": anchor_octave,
            "topk": topk,
        }
        self._csv_path = csv_path
        self._library = load_library()

    @property
    def library(self):
        return self._library

    def analyze(self, user_input: str, **overrides: Any) -> Dict[str, Any]:
        options = dict(self._defaults)
        options.update(overrides)
        return analyze(user_input, **options)


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
    metrics = best.get("metrics", {})
    if metrics.get("explanation"):
        lines.append(f"- Qualité du match : {metrics['explanation']} ({best['stars']}★)")
    alternatives = result.get("alternatives", [])
    if alternatives:
        lines.append("- Alternatives top-k :")
        for alt in alternatives:
            lines.append(f"  • {alt['stars']}★ {alt['copy_line']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys

    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Am"
    analysis = analyze(user_input)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print()
    print(format_analysis_fr(analysis))
