"""Noyau du traducteur Syntakt SY CHORD.

Ce module se charge de charger la bibliothèque officielle depuis ``Syntakt.csv``
ainsi que des règles de normalisation/scoring utilisées par ``Session``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import re

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "Syntakt.csv"

NOTE_PATTERN = re.compile(r"[A-Ga-g](?:#|b)?")

NOTE_VALUES: Dict[str, int] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

VALUE_TO_NOTE = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

BAL_MOTIFS = {
    74: "••..",
    84: "•••.",
    96: "••••",
    106: "•*••",
}

QUALITY_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "augmented": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "Maj7": (0, 4, 7, 11),
    "7": (0, 4, 7, 10),
    "m7": (0, 3, 7, 10),
    "mMaj7": (0, 3, 7, 11),
    "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),
    "M6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "augM7": (0, 4, 8, 11),
    "add9": (0, 4, 7, 14),
    "madd9": (0, 3, 7, 14),
}

QUALITY_ALIASES: Dict[str, str] = {
    "M": "major",
    "maj": "major",
    "major": "major",
    "M7": "Maj7",
    "Maj7": "Maj7",
    "maj7": "Maj7",
    "M6": "M6",
    "6": "M6",
    "maj6": "M6",
    "m": "minor",
    "min": "minor",
    "minor": "minor",
    "m7": "m7",
    "min7": "m7",
    "minor7": "m7",
    "7": "7",
    "dom7": "7",
    "dominant7": "7",
    "mMaj7": "mMaj7",
    "mM7": "mMaj7",
    "minorMaj7": "mMaj7",
    "minorMajor7": "mMaj7",
    "dim": "diminished",
    "diminished": "diminished",
    "dim7": "dim7",
    "o7": "dim7",
    "halfDim7": "m7b5",
    "ø7": "m7b5",
    "m7b5": "m7b5",
    "aug": "augmented",
    "augmented": "augmented",
    "augM7": "augM7",
    "+M7": "augM7",
    "add9": "add9",
    "Madd9": "add9",
    "madd9": "madd9",
    "sus2": "sus2",
    "sus4": "sus4",
}


@dataclass(frozen=True)
class Preset:
    name: str
    sy_type: str
    root: str
    intervals: Tuple[int, ...]
    bal: int
    bal_pattern: str
    labels: Tuple[str, ...]
    pcs: Tuple[int, ...]
    rel: Tuple[int, ...]
    inversion: Optional[int]

    @property
    def nvoices(self) -> int:
        return len(self.intervals)

    @property
    def bass_note(self) -> str:
        midi = (NOTE_VALUES[self.root] + self.intervals[0]) % 12
        return VALUE_TO_NOTE[midi]

    @property
    def motif(self) -> str:
        return BAL_MOTIFS.get(self.bal, self.bal_pattern)

    @property
    def notes_base(self) -> Tuple[str, ...]:
        return tuple(VALUE_TO_NOTE[(NOTE_VALUES[self.root] + off) % 12] for off in self.intervals)

    def voices_with_octaves(self, anchor_octave: int) -> List[Dict[str, object]]:
        base = midi_from_note_oct(self.root, anchor_octave)
        midis: List[int] = []
        for off in self.intervals:
            midis.append(base + off)
        midis.sort()
        out: List[Dict[str, object]] = []
        for midi in midis:
            note, octv = note_oct_from_midi(midi)
            out.append({
                "note": note,
                "octave": octv,
                "midi": midi,
                "label": f"{note}{octv}",
            })
        return out

    def as_dict(self, anchor_octave: int, *, stars: int, metrics: Dict[str, object], metrics_raw: Tuple[int, ...]) -> Dict[str, object]:
        voices = self.voices_with_octaves(anchor_octave)
        notes_oct = [v["label"] for v in voices]
        notes = [v["note"] for v in voices]
        span = voices[-1]["midi"] - voices[0]["midi"] if voices else 0
        copy_line = f"Root={self.root} Preset={self.sy_type} BAL={self.bal} ({self.motif})"
        return {
            "sy_root": self.root,
            "sy_type": self.sy_type,
            "preset_name": self.name,
            "bal": self.bal,
            "bal_motif": self.motif,
            "notes": notes,
            "notes_oct": notes_oct,
            "stars": stars,
            "metrics": metrics,
            "metrics_raw": list(metrics_raw),
            "voicing_span": span,
            "bal_preference_rank": metrics.get("bal_preference_rank", 9),
            "copy_line": copy_line,
        }


@dataclass(frozen=True)
class NormalizedInput:
    raw: str
    kind: str
    root: Optional[str]
    quality: Optional[str]
    notes: Tuple[str, ...]
    pcs_set: frozenset
    rel: Tuple[int, ...]
    inversion: Optional[int]
    bass: Optional[str]
    expected_quality: Optional[str]
    expected_intervals: Tuple[int, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "raw": self.raw,
            "kind": self.kind,
            "root": self.root,
            "quality": self.quality,
            "notes": list(self.notes),
            "pcs": sorted(self.pcs_set),
            "rel": list(self.rel),
            "inversion": self.inversion,
            "bass": self.bass,
            "expected_quality": self.expected_quality,
        }


def normalize_note(name: str) -> str:
    candidate = (name or "").strip()
    if candidate in NOTE_VALUES:
        value = NOTE_VALUES[candidate]
        return VALUE_TO_NOTE[value]
    upper = candidate.upper().replace("BEMOL", "B").replace("DIESE", "#")
    if upper in NOTE_VALUES:
        value = NOTE_VALUES[upper]
        return VALUE_TO_NOTE[value]
    raise ValueError(f"Note inconnue: {name}")


def midi_from_note_oct(name: str, octave: int) -> int:
    return 12 * (octave + 1) + NOTE_VALUES[name]


def note_oct_from_midi(midi: int) -> Tuple[str, int]:
    return VALUE_TO_NOTE[midi % 12], midi // 12 - 1


def _parse_intervals(raw: str) -> Tuple[int, ...]:
    values: List[int] = []
    for part in raw.split("|"):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Intervalle vide")
    return tuple(sorted(values))


def _compute_rel(intervals: Sequence[int]) -> Tuple[int, ...]:
    base = intervals[0]
    return tuple((x - base) % 12 for x in intervals)


def _detect_inversion(rel: Tuple[int, ...], sy_type: str) -> Optional[int]:
    expected = QUALITY_INTERVALS.get(sy_type)
    if not expected or len(expected) != len(rel):
        return None
    seq = list(expected)
    for inv in range(len(seq)):
        rotated = seq[inv:] + [x + 12 for x in seq[:inv]]
        rel_rot = tuple((x - rotated[0]) % 12 for x in rotated)
        if tuple(rel_rot) == tuple(rel):
            return inv
    return None


def _validate_row(row: Dict[str, str]) -> None:
    required = {"PresetName", "Type", "Root", "Intervals", "BAL", "BALPattern", "Labels"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"Colonnes manquantes dans Syntakt.csv: {', '.join(sorted(missing))}")
    root = row["Root"].strip()
    if root not in NOTE_VALUES:
        raise ValueError(f"Racine inconnue: {root}")
    if int(row["BAL"]) not in BAL_MOTIFS:
        raise ValueError(f"BAL non documenté: {row['BAL']}")


def _load_presets() -> List[Preset]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Fichier manquant: {CSV_PATH}")
    presets: List[Preset] = []
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            _validate_row(row)
            intervals = _parse_intervals(row["Intervals"])
            rel = _compute_rel(intervals)
            inversion = _detect_inversion(rel, row["Type"].strip())
            pcs = tuple(sorted({(NOTE_VALUES[row["Root"].strip()] + v) % 12 for v in intervals}))
            labels = tuple(part for part in row["Labels"].split(";") if part)
            preset = Preset(
                name=row["PresetName"].strip(),
                sy_type=row["Type"].strip(),
                root=normalize_note(row["Root"]),
                intervals=intervals,
                bal=int(row["BAL"]),
                bal_pattern=row["BALPattern"].strip(),
                labels=labels,
                pcs=pcs,
                rel=rel,
                inversion=inversion,
            )
            presets.append(preset)
    if not presets:
        raise ValueError("Aucun preset chargé depuis Syntakt.csv")
    return presets


@lru_cache(maxsize=1)
def load_library() -> List[Preset]:
    return _load_presets()


def _quality_from_alias(alias: str) -> Optional[str]:
    key = alias.strip()
    if not key:
        return "major"
    return QUALITY_ALIASES.get(key, None)


def _quality_from_pcs(pcs: Iterable[int], voice_count: int) -> Optional[str]:
    target = frozenset(pcs)
    for name, intervals in QUALITY_INTERVALS.items():
        if len(intervals) != voice_count:
            continue
        ref = frozenset({interval % 12 for interval in intervals})
        if ref == target:
            return name
    return None


def _guess_root_from_pcs(pcs: Sequence[int], quality: str) -> Optional[str]:
    if quality not in QUALITY_INTERVALS:
        return None
    offsets = QUALITY_INTERVALS[quality]
    for root_val in range(12):
        ref = { (root_val + off) % 12 for off in offsets }
        if ref == set(pcs):
            return VALUE_TO_NOTE[root_val]
    return None


def _relative_from_notes(notes: Sequence[str]) -> Tuple[int, ...]:
    if not notes:
        return tuple()
    values = [NOTE_VALUES[n] for n in notes]
    ordered: List[int] = []
    last = None
    for val in values:
        cur = val if last is None else val
        if last is not None:
            while cur <= last:
                cur += 12
        ordered.append(cur)
        last = cur
    base = ordered[0]
    return tuple((val - base) % 12 for val in ordered)


def normalize_input(user_input: str) -> NormalizedInput:
    text = (user_input or "").strip()
    if not text:
        raise ValueError("Entrée vide.")
    note_tokens = NOTE_PATTERN.findall(text)
    plain = re.sub(r"[\s,;-]", "", text)
    if note_tokens and len(note_tokens) >= 2 and plain.upper() == "".join(t.upper() for t in note_tokens):
        notes = tuple(normalize_note(tok) for tok in note_tokens)
        pcs = frozenset(NOTE_VALUES[n] for n in notes)
        rel = _relative_from_notes(notes)
        quality_guess = _quality_from_pcs(pcs, len(notes))
        root_guess = _guess_root_from_pcs(list(pcs), quality_guess) if quality_guess else None
        inversion = None
        if quality_guess:
            inversion = _detect_inversion(rel, quality_guess)
        return NormalizedInput(
            raw=text,
            kind="notes",
            root=root_guess,
            quality=quality_guess,
            notes=notes,
            pcs_set=pcs,
            rel=rel,
            inversion=inversion,
            bass=notes[0],
            expected_quality=quality_guess,
            expected_intervals=QUALITY_INTERVALS.get(quality_guess, tuple()),
        )
    match = re.match(r"^\s*([A-Ga-g](?:#|b)?)(.*)$", text)
    if not match:
        raise ValueError(
            "Entrée non reconnue. Indique un symbole d'accord (ex: Am, FMaj7) ou une liste de notes (ex: E A C)."
        )
    root = normalize_note(match.group(1))
    rest = (match.group(2) or "").strip()
    quality = _quality_from_alias(rest) if rest else "major"
    if not quality:
        raise ValueError(f"Qualité d'accord inconnue: {rest}")
    intervals = QUALITY_INTERVALS[quality]
    pcs = frozenset((NOTE_VALUES[root] + off) % 12 for off in intervals)
    rel = _compute_rel(intervals)
    return NormalizedInput(
        raw=text,
        kind="symbol",
        root=root,
        quality=quality,
        notes=tuple(),
        pcs_set=pcs,
        rel=rel,
        inversion=0,
        bass=root,
        expected_quality=quality,
        expected_intervals=intervals,
    )


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _explanation_from_stars(stars: int) -> str:
    if stars >= 5:
        return "exact"
    if stars >= 3:
        return "ensemble équivalent"
    if stars >= 1:
        return "voisinage"
    return "mismatch"


def _bal_preference(preset: Preset) -> int:
    if preset.nvoices == 3 and preset.rel == (0, 5, 9):
        if preset.bal == 106:
            return 0
        if preset.bal == 74:
            return 1
        return 2
    return 0


def rate_preset(
    preset: Preset,
    normalized: NormalizedInput,
    *,
    strict: bool,
    pref_bass_root: bool,
    pref_root_ident: bool,
) -> Optional[Tuple[int, Dict[str, object], Tuple[int, ...]]]:
    target_pcs = normalized.pcs_set
    candidate_pcs = set(preset.pcs)
    pitch_match = candidate_pcs == target_pcs if target_pcs else False
    rel_match = normalized.rel == preset.rel and bool(normalized.rel)
    same_voicing = pitch_match and rel_match
    bass_match = normalized.bass == preset.bass_note if normalized.bass else False
    root_match = normalized.root == preset.root if normalized.root else False
    missing = len(target_pcs - candidate_pcs)
    extra = len(candidate_pcs - target_pcs)
    j = jaccard(target_pcs, candidate_pcs)
    stars: int
    if same_voicing:
        stars = 5
    elif pitch_match:
        stars = 4 if (root_match or normalized.kind == "symbol") else 3
    elif j >= 0.75:
        stars = 2
    elif j >= 0.5:
        stars = 1
    else:
        stars = 0
    if strict and stars < 3:
        return None
    bal_rank = _bal_preference(preset)
    set_distance = missing + extra
    explanation = _explanation_from_stars(stars)
    metrics = {
        "same_voicing": same_voicing,
        "root_match": root_match,
        "bass_match": bass_match,
        "bal_preference_rank": bal_rank,
        "missing": missing,
        "extra": extra,
        "set_distance": set_distance,
        "jaccard": round(j, 4),
        "explanation": explanation,
    }
    inversion_penalty = 0
    if normalized.inversion is not None and preset.inversion is not None:
        inversion_penalty = abs(normalized.inversion - preset.inversion)
    root_penalty = 0 if root_match else 1 if normalized.root else 0
    bass_penalty = 0 if bass_match else 1 if normalized.bass else 0
    if pref_bass_root and preset.bass_note == preset.root:
        bass_penalty = max(bass_penalty - 1, 0)
    if pref_root_ident and root_match:
        root_penalty = 0
    metrics_raw = (
        inversion_penalty,
        root_penalty,
        bass_penalty,
        bal_rank,
        -stars,
        -round(j * 1000),
    )
    return stars, metrics, metrics_raw


def rank_presets(
    normalized: NormalizedInput,
    *,
    strict: bool,
    pref_bass_root: bool,
    pref_root_ident: bool,
) -> List[Tuple[Preset, int, Dict[str, object], Tuple[int, ...]]]:
    library = load_library()
    scored: List[Tuple[Preset, int, Dict[str, object], Tuple[int, ...]]] = []
    for preset in library:
        rating = rate_preset(
            preset,
            normalized,
            strict=strict,
            pref_bass_root=pref_bass_root,
            pref_root_ident=pref_root_ident,
        )
        if rating is None:
            continue
        stars, metrics, metrics_raw = rating
        scored.append((preset, stars, metrics, metrics_raw))
    scored.sort(key=lambda item: (item[3], -item[1], item[0].name))
    return scored


ENGINE_VERSION = "3.x"
SCHEMA_VERSION = "2025.05"
ANALYSIS_JSON_SCHEMA = {
    "title": "syntakt.analysis",
    "version": SCHEMA_VERSION,
    "description": "Structure des résultats retournés par Session.analyze",
}
