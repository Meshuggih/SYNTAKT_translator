"""Core logic for Syntakt chord analysis, independent from Pythonista APIs."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import json
import logging
import math
import re

from .kb_scales import recommend_kb_scale

__all__ = [
    "Session",
    "AnalysisResult",
    "MatchCandidate",
    "format_analysis_fr",
    "aggregated_chord_pcs_from_results",
    "recommend_kb_scale",
    "format_for_syntakt",
    "normalize_input",
    "get_doc",
    "get_version",
    "ANALYSIS_JSON_SCHEMA",
    "NOTE_VALUES",
    "VALUES_TO_NOTES",
    "ALL_ROOTS",
    "normalize_note",
    "midi_from_note_oct",
    "note_oct_from_midi",
    "build_syntakt_library",
    "voices_with_octaves",
    "parse_free_input",
    "list_matches_symbol",
    "list_matches_notes",
]


log = logging.getLogger("sychord")

ENGINE_VERSION = "SYCHORD-2025.10"
SCHEMA_VERSION = "1.0"

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "Syntakt.csv"
DOC_PATH = DATA_DIR / "syntakt_documentation.json"

EXPECTED_CSV_COLUMNS = {
    "sy_root",
    "sy_type",
    "bal",
    "bal_pattern",
    "intervals",
}

_DOC_CACHE: Optional[Dict[str, Any]] = None


def _validate_csv_dataset(path: Path = CSV_PATH) -> None:
    """Validate the optional CSV dataset structure when present."""

    if not path.exists():
        log.debug("CSV dataset %s not found – skipping validation", path)
        return
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers = set(reader.fieldnames or [])
            missing = sorted(EXPECTED_CSV_COLUMNS - headers)
            if missing:
                log.error(
                    "Colonnes manquantes dans %s: %s",
                    path,
                    ", ".join(missing),
                )
            else:
                # Touch the iterator to surface malformed rows early without
                # loading the entire file.
                next(reader, None)
    except Exception as exc:  # pragma: no cover - defensive safeguard
        log.error("Validation du dataset %s impossible: %s", path, exc)


def _load_doc_cache(path: Path = DOC_PATH) -> Dict[str, Any]:
    """Load and cache the optional documentation JSON file."""

    global _DOC_CACHE
    if _DOC_CACHE is not None:
        return _DOC_CACHE
    if not path.exists():
        log.debug("Documentation file %s not found", path)
        _DOC_CACHE = {}
        return _DOC_CACHE
    try:
        _DOC_CACHE = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        log.error("Lecture impossible de %s: %s", path, exc)
        _DOC_CACHE = {}
    return _DOC_CACHE

# ---------------------- Notes & Normalisation ----------------------
NOTE_VALUES: Dict[str, int] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "E#": 5,
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
    "B#": 0,
}

# affichage préférant dièses
VALUES_TO_NOTES: Dict[int, str] = {}
for k, v in NOTE_VALUES.items():
    if v not in VALUES_TO_NOTES:
        VALUES_TO_NOTES[v] = k
for k, v in [
    ("C", 0),
    ("C#", 1),
    ("D", 2),
    ("D#", 3),
    ("E", 4),
    ("F", 5),
    ("F#", 6),
    ("G", 7),
    ("G#", 8),
    ("A", 9),
    ("A#", 10),
    ("B", 11),
]:
    VALUES_TO_NOTES[v] = k

ALL_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def normalize_note(name: str) -> str:
    """Normalise a note name by stripping and favouring sharps."""
    name = (name or "").strip()
    if name in NOTE_VALUES:
        return VALUES_TO_NOTES[NOTE_VALUES[name]]
    u = name.upper().replace("BEMOL", "B").replace("DIESE", "#")
    if u in NOTE_VALUES:
        return VALUES_TO_NOTES[NOTE_VALUES[u]]
    raise ValueError(f"Note invalide: {name}")


def midi_from_note_oct(name: str, octv: int) -> int:
    """Return the MIDI value for a note name + octave (C-1 == 0)."""
    return 12 * (octv + 1) + NOTE_VALUES[name]


def note_oct_from_midi(m: int) -> Tuple[str, int]:
    return VALUES_TO_NOTES[m % 12], (m // 12) - 1


# ---------------------- Chords & Presets ----------------------
REAL_TRIADS: Dict[str, List[int]] = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}
REAL_TETRADS: Dict[str, List[int]] = {
    "major7": [0, 4, 7, 11],
    "minor7": [0, 3, 7, 10],
    "dominant7": [0, 4, 7, 10],
    "minorMajor7": [0, 3, 7, 11],
    "diminished7": [0, 3, 6, 9],
    "halfDim7": [0, 3, 6, 10],
    "major6": [0, 4, 7, 9],
    "minor6": [0, 3, 7, 9],
    "augMajor7": [0, 4, 8, 11],
    "add9_major": [0, 4, 7, 14],
    "add9_minor": [0, 3, 7, 14],
}
REAL_SYNONYMS: Dict[str, str] = {
    "maj": "major",
    "min": "minor",
    "dim": "diminished",
    "aug": "augmented",
    "M7": "major7",
    "maj7": "major7",
    "m7": "minor7",
    "min7": "minor7",
    "7": "dominant7",
    "dom7": "dominant7",
    "mMaj7": "minorMajor7",
    "mmaj7": "minorMajor7",
    "minMaj7": "minorMajor7",
    "dim7": "diminished7",
    "m7b5": "halfDim7",
    "ø7": "halfDim7",
    "M6": "major6",
    "6": "major6",
    "m6": "minor6",
    "augM7": "augMajor7",
    "add9": "add9_major",
    "madd9": "add9_minor",
}

CHORD_INTERVALS: Dict[str, List[int]] = {
    # --- Unisons ---
    "unison1": [0],
    "unison2": [0, 0],
    "unison3": [0, 0, 0],
    "unison4": [0, 0, 0, 0],
    # --- Dyades ---
    "Fourth": [0, 5],
    "Fifth": [0, 7],
    # --- Triades ---
    "minor": [0, 3, 7],
    "Major": [0, 4, 7],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    # --- Tétra des ---
    "m7": [0, 3, 7, 10],
    "M7": [0, 4, 7, 10],
    "Maj7": [0, 4, 7, 11],
    "mMaj7": [0, 3, 7, 11],
    "7sus4": [0, 5, 7, 10],
    "dim7": [0, 3, 6, 9],
    # --- Variantes ---
    "madd9": [0, 3, 7, 14],
    "Madd9": [0, 4, 7, 14],
    "m6": [0, 3, 7, 9],
    "M6": [0, 4, 7, 9],
    "mb5": [0, 3, 6],
    "Mb5": [0, 4, 6],
    "m7b5": [0, 3, 6, 10],
    "M7b5": [0, 4, 6, 10],
    "M#5": [0, 4, 8],
    "m7#5": [0, 3, 8, 10],
    "M7#5": [0, 4, 8, 10],
    "mb6": [0, 3, 8],
    "m9no5": [0, 3, 10, 14],
    "M9no5": [0, 4, 10, 14],
    "Madd9b5": [0, 4, 6, 14],
    "Maj7b5": [0, 4, 6, 11],
    "M7b9no5": [0, 4, 10, 13],
    "sus4#5b9": [0, 1, 5, 8],
    "sus4add#5": [0, 5, 7, 8],
    "Maddb5": [0, 4, 6, 7],
    "M6add4no5": [0, 4, 5, 9],
    "Maj7/6no5": [0, 4, 9],
    "Maj9no5": [0, 4, 11, 14],
}

BAL_PATTERNS: Dict[int, str] = {
    0: "•~~~",
    10: "••~~",
    20: "•••~",
    30: "••••",
    32: "root",
    37: "•~••",
    42: "•.••",
    47: "•.~•",
    52: "•..•",
    57: "•..~",
    62: "•...",
    69: "•~..",
    74: "••..",
    79: "••~.",
    84: "•••.",
    91: "•••~",
    96: "••••",
    101: "•~••",
    106: "•*••",
    111: "•*~•",
    116: "•**•",
    121: "•**~",
    127: "•***",
}


def _pattern_to_oct(pattern: str, nvoices: int) -> List[Optional[int]]:
    if pattern == "root":
        pattern = "••••"
    pat = pattern.replace("...", "..").replace("...", "..")
    pat = (pat + "~~~~")[: max(1, min(4, nvoices))]
    out: List[Optional[int]] = []
    for ch in pat:
        if ch == "•":
            out.append(0)
        elif ch == ".":
            out.append(-12)
        elif ch == "*":
            out.append(12)
        elif ch == "~":
            out.append(None)
        else:
            out.append(None)
    while len(out) < nvoices:
        out.append(None)
    return out[:nvoices]


def _octify_extension_offsets(sy_type: str, offsets: Sequence[int]) -> List[int]:
    """Ensure extensions (9/11/13) are voiced above the octave when requested."""

    lowered = sy_type.lower()
    if not any(tag in lowered for tag in ("9", "11", "13")):
        return list(offsets)

    extension_classes = {
        "9": {1, 2, 3},  # b9, 9, #9
        "11": {5, 6},  # 11, #11
        "13": {8, 9, 10},  # b13, 13, #13
    }

    adjusted: List[int] = []
    for idx, off in enumerate(offsets):
        new_off = off
        if off < 12 and idx >= 3:
            for tag, pcs in extension_classes.items():
                if tag in lowered and (off % 12) in pcs:
                    new_off = off + 12
                    break
        adjusted.append(new_off)
    return adjusted


# ---------------------- Library build helpers ----------------------

def build_syntakt_library(*, octify_extensions: bool = True) -> List[Dict[str, Any]]:
    """Build the exhaustive library of Syntakt presets (pure, deterministic)."""
    lib: List[Dict[str, Any]] = []
    for sy_root in ALL_ROOTS:
        rv = NOTE_VALUES[sy_root]
        for sy_type, offsets in CHORD_INTERVALS.items():
            adj_offsets = _octify_extension_offsets(sy_type, offsets) if octify_extensions else list(offsets)
            nvoices = min(4, len(adj_offsets))
            base_offsets = adj_offsets[:nvoices]
            for bal, pat in BAL_PATTERNS.items():
                octs = _pattern_to_oct(pat, nvoices)
                abs_semi: List[int] = []
                for i, off in enumerate(base_offsets):
                    if i >= len(octs) or octs[i] is None:
                        continue
                    abs_semi.append(rv + off + octs[i])
                if not abs_semi:
                    continue
                abs_semi.sort()
                pcs = sorted({s % 12 for s in abs_semi})
                base = abs_semi[0]
                rel = [(s - base) % 12 for s in abs_semi]
                notes = [VALUES_TO_NOTES[s % 12] for s in abs_semi]
                lib.append(
                    {
                        "sy_root": sy_root,
                        "sy_type": sy_type,
                        "bal": bal,
                        "bal_pattern": pat,
                        "intervals": base_offsets,
                        "octs": octs,
                        "notes": notes,
                        "pcs_set": pcs,
                        "rel_order": rel,
                        "root_pc": base % 12,
                        "nvoices": len(notes),
                    }
                )
    return lib


# Preferences for inversions (triads)
PREF_BAL_BY_TRIAD_REL = {
    (0, 4, 7): [20, 30, 91],
    (0, 3, 8): [79, 30, 20],
    (0, 5, 9): [106, 74, 30, 20],
    (0, 3, 7): [20, 30, 91],
    (0, 4, 9): [79, 30, 20],
    (0, 5, 8): [106, 74, 30, 20],
}

TRIAD_INVERSION_MAP: Dict[Tuple[int, ...], int] = {}
for offsets in REAL_TRIADS.values():
    seq = list(offsets)
    for inversion, base in enumerate(seq):
        rotated = tuple(sorted(((o - base) % 12) for o in seq))
        TRIAD_INVERSION_MAP.setdefault(rotated, inversion)


def triad_bal_pref_rank(c: Dict[str, Any]) -> int:
    if c.get("nvoices") != 3:
        return 9
    key = tuple(c.get("rel_order") or [])
    prefs = PREF_BAL_BY_TRIAD_REL.get(key)
    if not prefs:
        return 9
    try:
        return prefs.index(c["bal"])
    except ValueError:
        return 8


# ---------------------- Octave handling ----------------------

def voices_with_octaves(cand: Dict[str, Any], anchor_octave: int = 3) -> List[Dict[str, Any]]:
    root = cand["sy_root"]
    intervals = cand["intervals"]
    octs = cand["octs"]
    n = min(len(intervals), 4)
    base_midi = midi_from_note_oct(root, anchor_octave)
    mids = []
    for i in range(n):
        if i >= len(octs) or octs[i] is None:
            continue
        mids.append(base_midi + intervals[i] + octs[i])
    mids.sort()
    out = []
    for m in mids:
        nname, octv = note_oct_from_midi(m)
        out.append({"note": nname, "octave": octv, "midi": m, "label": f"{nname}{octv}"})
    return out


def _notes_oct_from_sequence(notes: Sequence[str], anchor_octave: int) -> List[str]:
    labels: List[str] = []
    prev_midi: Optional[int] = None
    current_oct = anchor_octave
    for note in notes:
        midi_val = midi_from_note_oct(note, current_oct)
        while prev_midi is not None and midi_val <= prev_midi:
            current_oct += 1
            midi_val = midi_from_note_oct(note, current_oct)
        nname, octv = note_oct_from_midi(midi_val)
        labels.append(f"{nname}{octv}")
        prev_midi = midi_val
    return labels


# ---------------------- Parsing ----------------------
NOTE_TOKEN = r"[A-Ga-g](?:#|b)?"


def parse_free_input(user_input: str) -> Tuple[str, Any]:
    s = (user_input or "").strip()
    if not s:
        raise ValueError("Entrée vide.")
    cleaned = s.replace("-", " ").replace(",", " ").replace(";", " ")
    tokens = re.findall(NOTE_TOKEN, cleaned)
    if tokens:
        cat = "".join(t.upper() for t in tokens)
        only_notes = re.sub(r"[^A-Ga-g#b]", "", s).upper()
        if cat == only_notes and len(tokens) >= 2:
            return "notes", [normalize_note(t) for t in tokens]
    m = re.match(r"^\s*(" + NOTE_TOKEN + r")\s*(.*)$", s)
    if m:
        root = normalize_note(m.group(1))
        qual = (m.group(2) or "major").strip()
        qual = (
            qual.replace("minor", "m")
            .replace("major", "M")
            .replace("min", "m")
            .replace("maj", "M")
        )
        if qual == "m":
            qual = "minor"
        elif qual == "M":
            qual = "major"
        if qual in REAL_SYNONYMS:
            qual = REAL_SYNONYMS[qual]
        if qual in REAL_TRIADS or qual in REAL_TETRADS:
            return "symbol", (root, qual)
    raise ValueError(
        "Entrée non reconnue. Donne un nom d’accord (ex: Am, FMaj7) OU des notes graves→aigu (ex: E A C)."
    )


# ---------------------- Scoring helpers ----------------------

def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(a)
    sb = set(b)
    return len(sa & sb) / float(len(sa | sb)) if (sa or sb) else 1.0


def rel_from_note_list(notes_low_to_high: Sequence[str]) -> List[int]:
    if not notes_low_to_high:
        return []
    base = NOTE_VALUES[normalize_note(notes_low_to_high[0])]
    rel: List[int] = []
    prev = base
    for n in notes_low_to_high:
        pc = NOTE_VALUES[normalize_note(n)]
        k = 0
        while (pc + 12 * k) < prev:
            k += 1
        rel.append((pc + 12 * k - base) % 12)
        prev = pc + 12 * k
    return sorted(rel)


def real_chord_offsets(quality: str) -> List[int]:
    q = (quality or "").strip()
    if q in REAL_SYNONYMS:
        q = REAL_SYNONYMS[q]
    if q in REAL_TRIADS:
        return REAL_TRIADS[q]
    if q in REAL_TETRADS:
        return REAL_TETRADS[q]
    raise ValueError(f"Qualité d’accord inconnue: {quality}")


def real_chord_pcs_and_order(root: str, quality: str) -> Tuple[List[int], List[int], str, int]:
    r = normalize_note(root)
    r_val = NOTE_VALUES[r]
    offs = real_chord_offsets(quality)
    abs_vals = sorted(r_val + o for o in offs)
    pcs = sorted({v % 12 for v in abs_vals})
    base = abs_vals[0]
    rel = [(v - base) % 12 for v in abs_vals]
    return pcs, rel, r, r_val


MetricKey = Tuple[int, int, int, int, int, int, int, str]


def rate_symbol_candidate(
    c: Dict[str, Any],
    root: str,
    quality: str,
    real_pcs: Sequence[int],
    real_rel: Sequence[int],
) -> MetricKey:
    inv = 0 if c["rel_order"] == list(real_rel) else 1
    root_match = 0 if c["sy_root"] == root else 1
    bass_match = 0 if c["root_pc"] == NOTE_VALUES[root] else 1
    bal_pref = triad_bal_pref_rank(c)
    cset = set(c["pcs_set"])
    rset = set(real_pcs)
    miss = len(rset - cset)
    extra = len(cset - rset)
    set_dist = 0 if (miss == 0 and extra == 0) else (10 - int(10 * jaccard(cset, rset)))
    if miss == 0 and extra == 0 and inv == 0 and root_match == 0:
        reason = "exact"
    elif miss == 0 and extra == 0 and inv == 0:
        reason = "exact-ordre"
    elif miss == 0 and extra == 0:
        reason = "mêmes notes (inversion)"
    else:
        reason = "approx (ensemble diff.)"
    return (inv, root_match, bass_match, bal_pref, miss, extra, set_dist, reason)


def stars_from_symbol_metrics(
    metrics: MetricKey, pref_bass: bool = True, pref_root: bool = True
) -> int:
    inv, root_match, bass_match, _bal_pref, miss, extra, set_dist, _ = metrics
    if miss == 0 and extra == 0 and inv == 0 and root_match == 0:
        return 5
    if miss == 0 and extra == 0 and inv == 0:
        return 4
    if miss == 0 and extra == 0:
        if (pref_bass and bass_match) or (pref_root and root_match):
            return 3
        return 4
    if set_dist <= 3:
        return 2
    return 1


def rate_notes_candidate(c: Dict[str, Any], user_notes: Sequence[str]) -> MetricKey:
    user_rel = rel_from_note_list(user_notes)
    user_set = sorted({NOTE_VALUES[n] for n in user_notes})
    inv = 0 if c["rel_order"] == user_rel else 1
    cset = c["pcs_set"]
    miss = len(set(user_set) - set(cset))
    extra = len(set(cset) - set(user_set))
    set_dist = 0 if (miss == 0 and extra == 0) else (10 - int(10 * jaccard(user_set, cset)))
    bal_pref = triad_bal_pref_rank(c)
    bass_pc = NOTE_VALUES[normalize_note(user_notes[0])]
    bass_match = 0 if c["root_pc"] == bass_pc else 1
    root_match = 0
    reason = (
        "exact"
        if (miss == 0 and extra == 0 and inv == 0)
        else ("mêmes notes (inversion)" if (miss == 0 and extra == 0) else "approx (ensemble diff.)")
    )
    return (inv, root_match, bass_match, bal_pref, miss, extra, set_dist, reason)


def stars_from_notes_metrics(metrics: MetricKey, pref_bass: bool = True) -> int:
    inv, _root_match, bass_match, _bal_pref, miss, extra, set_dist, _ = metrics
    if miss == 0 and extra == 0 and inv == 0:
        return 5
    if miss == 0 and extra == 0:
        return 4 if (pref_bass and bass_match) else 5
    if set_dist <= 3:
        return 2
    return 1


# ---------------------- Matching ----------------------

def list_matches_symbol(
    root: str,
    quality: str,
    sy_lib: Sequence[Dict[str, Any]],
    anchor_octave: int,
    topk: int,
    strict: bool,
    pref_bass: bool,
    pref_root: bool,
) -> List["MatchCandidate"]:
    rpcs, rrel, rr_norm, rr_val = real_chord_pcs_and_order(root, quality)
    rr_pc = rr_val % 12
    cands = [c for c in sy_lib if set(c["pcs_set"]) == set(rpcs)]
    if strict:
        cands = [c for c in cands if c["rel_order"] == list(rrel)]

    def _key(c: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        inv = 0 if c["rel_order"] == list(rrel) else 1
        root_match = 0 if c["sy_root"] == rr_norm else (2 if pref_root else 1)
        bass_match = 0 if c["root_pc"] == rr_pc else (2 if pref_bass else 1)
        bal_pref = triad_bal_pref_rank(c)
        return (inv, root_match, bass_match, bal_pref, -c.get("nvoices", 4))

    cands = sorted(cands, key=_key)
    out: List[MatchCandidate] = []
    for c in cands[:64]:
        metrics = rate_symbol_candidate(c, rr_norm, quality, rpcs, rrel)
        stars = stars_from_symbol_metrics(metrics, pref_bass, pref_root)
        out.append(
            _build_match_candidate(
                c,
                metrics=metrics,
                stars=stars,
                anchor_octave=anchor_octave,
                query_root=rr_norm,
                query_quality=quality,
            )
        )
    out.sort(key=_candidate_sort_key)
    return out[:topk]


def list_matches_notes(
    user_notes: Sequence[str],
    sy_lib: Sequence[Dict[str, Any]],
    anchor_octave: int,
    topk: int,
    strict: bool,
    pref_bass: bool,
) -> List["MatchCandidate"]:
    user_set = sorted({NOTE_VALUES[n] for n in user_notes})
    user_rel = rel_from_note_list(user_notes)
    second_inversion = TRIAD_INVERSION_MAP.get(tuple(user_rel)) == 2
    cands = [c for c in sy_lib if set(c["pcs_set"]) == set(user_set)]
    if strict:
        cands = [c for c in cands if c["rel_order"] == user_rel]

    def _key(c: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        inv = 0 if c["rel_order"] == user_rel else 1
        bass_pc = NOTE_VALUES[user_notes[0]]
        bass_match = 0 if c["root_pc"] == bass_pc else (2 if pref_bass else 1)
        bal_pref = triad_bal_pref_rank(c)
        bal_bias = 0
        if second_inversion and c.get("nvoices") == 3:
            if c.get("bal") == 106:
                inv = 0  # treat as preferred match
                bass_match = 0
                bal_bias = -1
                bal_pref = 0
            elif c.get("bal") == 74:
                bal_bias = 1
        return (inv, bass_match, bal_bias, bal_pref, -c.get("nvoices", 4))

    cands = sorted(cands, key=_key)
    out: List[MatchCandidate] = []
    for c in cands[:64]:
        metrics = rate_notes_candidate(c, user_notes)
        stars = stars_from_notes_metrics(metrics, pref_bass=pref_bass)
        out.append(
            _build_match_candidate(
                c,
                metrics=metrics,
                stars=stars,
                anchor_octave=anchor_octave,
                user_notes=user_notes,
            )
        )
    out.sort(key=_candidate_sort_key)
    return out[:topk]


# ---------------------- Session API ----------------------

def _metrics_to_dict(metrics: MetricKey) -> Dict[str, Any]:
    inv, root_match, bass_match, bal_pref, miss, extra, set_dist, reason = metrics
    return {
        "same_voicing": inv == 0,
        "root_match": root_match == 0,
        "bass_match": bass_match == 0,
        "bal_pref_rank": bal_pref,
        "missing": miss,
        "extra": extra,
        "set_distance": set_dist,
        "explanation": reason,
    }


def _format_copy_line(root: str, preset: str, bal: int, motif: str) -> str:
    motif_text = motif or ""
    return f"Root={root} Preset={preset} BAL={bal} ({motif_text})"


@dataclass(frozen=True)
class MatchCandidate:
    """Immutable representation of a ranked Syntakt preset."""

    sy_root: str
    sy_type: str
    bal: int
    bal_motif: str
    notes: List[str] = field(default_factory=list)
    notes_oct: List[str] = field(default_factory=list)
    stars: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    metrics_raw: Tuple[int, ...] = field(default_factory=tuple)
    voicing_span: int = 0
    bal_preference_rank: int = 9
    copy_line: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "sy_root": self.sy_root,
            "sy_type": self.sy_type,
            "bal": self.bal,
            "bal_motif": self.bal_motif,
            "notes": list(self.notes),
            "notes_oct": list(self.notes_oct),
            "stars": self.stars,
            "metrics": dict(self.metrics),
            "metrics_raw": list(self.metrics_raw),
            "voicing_span": self.voicing_span,
            "bal_preference_rank": self.bal_preference_rank,
            "copy_line": self.copy_line,
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MatchCandidate":
        extra = dict(payload)
        sy_root = extra.pop("sy_root")
        sy_type = extra.pop("sy_type")
        bal = int(extra.pop("bal"))
        bal_motif = extra.pop("bal_motif", extra.pop("bal_pattern", ""))
        notes = list(extra.pop("notes", []))
        notes_oct = list(extra.pop("notes_oct", []))
        stars = int(extra.pop("stars", 0))
        metrics = dict(extra.pop("metrics", {}))
        metrics_raw = tuple(extra.pop("metrics_raw", ()))
        voicing_span = int(extra.pop("voicing_span", 0))
        bal_preference_rank = int(extra.pop("bal_preference_rank", 9))
        copy_line = extra.pop("copy_line", _format_copy_line(sy_root, sy_type, bal, bal_motif))
        return cls(
            sy_root=sy_root,
            sy_type=sy_type,
            bal=bal,
            bal_motif=bal_motif,
            notes=notes,
            notes_oct=notes_oct,
            stars=stars,
            metrics=metrics,
            metrics_raw=metrics_raw,
            voicing_span=voicing_span,
            bal_preference_rank=bal_preference_rank,
            copy_line=copy_line,
            extra=extra,
        )


def _build_match_candidate(
    base: Dict[str, Any],
    *,
    metrics: MetricKey,
    stars: int,
    anchor_octave: int,
    query_root: Optional[str] = None,
    query_quality: Optional[str] = None,
    user_notes: Optional[Sequence[str]] = None,
) -> MatchCandidate:
    voices = voices_with_octaves(base, anchor_octave=anchor_octave)
    notes_oct = [v["label"] for v in voices]
    override_applied = False
    span = voices[-1]["midi"] - voices[0]["midi"] if len(voices) > 1 else 0
    bal_motif = base.get("bal_pattern") or base.get("bal_motif") or ""
    override_metrics = metrics
    override_stars = stars
    if user_notes is not None:
        user_rel = rel_from_note_list(user_notes)
        if TRIAD_INVERSION_MAP.get(tuple(user_rel)) == 2 and base.get("bal") == 106 and base.get("nvoices") == 3:
            override_metrics = (0, 0, 0, 0, 0, 0, 0, "exact (BAL 106 priorisé)")
            override_stars = max(stars, 5)
            notes_oct = _notes_oct_from_sequence(user_notes, anchor_octave)
            override_applied = True
    metrics_dict = _metrics_to_dict(override_metrics)
    bal_pref_rank = triad_bal_pref_rank(base)
    if override_applied:
        bal_pref_rank = 0
    extra: Dict[str, Any] = {
        "intervals": list(base.get("intervals", [])),
        "octs": list(base.get("octs", [])),
        "nvoices": base.get("nvoices", len(notes_oct)),
        "rel_order": list(base.get("rel_order", [])),
        "pcs_set": list(base.get("pcs_set", [])),
        "query_root": query_root,
        "query_quality": query_quality,
        "user_notes": list(user_notes) if user_notes is not None else None,
    }
    extra = {k: v for k, v in extra.items() if v is not None}
    return MatchCandidate(
        sy_root=base["sy_root"],
        sy_type=base["sy_type"],
        bal=base["bal"],
        bal_motif=bal_motif,
        notes=list(base.get("notes", [])),
        notes_oct=notes_oct,
        stars=override_stars,
        metrics=metrics_dict,
        metrics_raw=tuple(override_metrics),
        voicing_span=span,
        bal_preference_rank=bal_pref_rank,
        copy_line=_format_copy_line(base["sy_root"], base["sy_type"], base["bal"], bal_motif),
        extra=extra,
    )


def _candidate_sort_key(candidate: MatchCandidate) -> Tuple[Any, ...]:
    prefer_106 = 0
    if candidate.extra.get("nvoices") == 3 and candidate.metrics_raw:
        if candidate.bal == 106:
            prefer_106 = -1
        elif candidate.bal == 74:
            prefer_106 = 1
    quality_rank = 0
    query_quality = candidate.extra.get("query_quality")
    if query_quality and candidate.sy_type != query_quality:
        quality_rank = 1
    return (
        -candidate.stars,
        tuple(candidate.metrics_raw),
        prefer_106,
        quality_rank,
        candidate.bal_preference_rank,
        candidate.voicing_span,
        candidate.sy_type,
        candidate.copy_line,
    )


ANALYSIS_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "sychord.analysis",
    "type": "object",
    "required": [
        "schema_version",
        "engine_version",
        "query",
        "best",
        "alternatives",
        "copy_lines",
        "error",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "engine_version": {"type": "string"},
        "query": {"type": "object"},
        "best": {"type": ["object", "null"]},
        "alternatives": {"type": "array"},
        "copy_lines": {"type": "array", "items": {"type": "string"}},
        "error": {"type": ["string", "null"]},
    },
}


@dataclass(frozen=True)
class AnalysisResult:
    query: Dict[str, Any]
    best: Optional[MatchCandidate]
    alternatives: List[MatchCandidate] = field(default_factory=list)
    copy_lines: List[str] = field(default_factory=list)
    error: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    engine_version: str = ENGINE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "query": self.query,
            "best": self.best.to_dict() if self.best else None,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "copy_lines": list(self.copy_lines),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnalysisResult":
        best_payload = payload.get("best")
        best = MatchCandidate.from_dict(best_payload) if best_payload else None
        alternatives = [MatchCandidate.from_dict(alt) for alt in payload.get("alternatives", [])]
        return cls(
            query=dict(payload.get("query", {})),
            best=best,
            alternatives=alternatives,
            copy_lines=list(payload.get("copy_lines", [])),
            error=payload.get("error"),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
            engine_version=payload.get("engine_version", ENGINE_VERSION),
        )


class Session:
    """Headless analysis session for Syntakt chord matching."""

    def __init__(
        self,
        library: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        octify_extensions: bool = True,
    ) -> None:
        _validate_csv_dataset()
        self.library = (
            list(library)
            if library is not None
            else build_syntakt_library(octify_extensions=octify_extensions)
        )

        @lru_cache(maxsize=256)
        def _cached(
            normalized_input: str,
            strict: bool,
            anchor_octave: int,
            topk: int,
            pref_bass_root: bool,
            pref_root_ident: bool,
        ) -> AnalysisResult:
            return self._perform_analysis(
                normalized_input,
                strict=strict,
                anchor_octave=anchor_octave,
                topk=topk,
                pref_bass_root=pref_bass_root,
                pref_root_ident=pref_root_ident,
            )

        self._cached_analyze = _cached

    def analyze(
        self,
        input_text: str,
        *,
        strict: bool = False,
        anchor_octave: int = 3,
        topk: int = 12,
        pref_bass_root: bool = True,
        pref_root_ident: bool = False,
    ) -> Dict[str, Any]:
        """Analyse ``input_text`` and return a structured dictionary."""

        normalized_input = input_text.strip()
        result = self._cached_analyze(
            normalized_input,
            strict,
            anchor_octave,
            topk,
            pref_bass_root,
            pref_root_ident,
        )
        # Refresh the query with the original raw string (without impacting cache).
        refreshed_query = dict(result.query)
        refreshed_query["raw"] = input_text
        refreshed = AnalysisResult(
            query=refreshed_query,
            best=result.best,
            alternatives=result.alternatives,
            copy_lines=result.copy_lines,
            error=result.error,
            schema_version=result.schema_version,
            engine_version=result.engine_version,
        )
        return refreshed.to_dict()

    def _perform_analysis(
        self,
        input_text: str,
        *,
        strict: bool,
        anchor_octave: int,
        topk: int,
        pref_bass_root: bool,
        pref_root_ident: bool,
    ) -> AnalysisResult:
        query_info: Dict[str, Any] = {
            "raw": input_text,
            "strict": strict,
            "anchor_octave": anchor_octave,
            "topk": topk,
            "pref_bass_root": pref_bass_root,
            "pref_root_ident": pref_root_ident,
        }
        try:
            kind, payload = parse_free_input(input_text)
            query_info["kind"] = kind
            if kind == "symbol":
                root, quality = payload
                query_info.update({"root": root, "quality": quality})
                alts = list_matches_symbol(
                    root,
                    quality,
                    self.library,
                    anchor_octave=anchor_octave,
                    topk=topk,
                    strict=strict,
                    pref_bass=pref_bass_root,
                    pref_root=pref_root_ident,
                )
            else:
                notes = list(payload)
                query_info["notes"] = notes
                alts = list_matches_notes(
                    notes,
                    self.library,
                    anchor_octave=anchor_octave,
                    topk=topk,
                    strict=strict,
                    pref_bass=pref_bass_root,
                )
        except ValueError as exc:
            return AnalysisResult(
                query=query_info,
                best=None,
                alternatives=[],
                copy_lines=[],
                error=str(exc),
            )

        copy_lines = [cand.copy_line for cand in alts] or []
        best = alts[0] if alts else None
        return AnalysisResult(query=query_info, best=best, alternatives=list(alts), copy_lines=copy_lines)


# ---------------------- Keyboard advisor helpers ----------------------

def aggregated_chord_pcs_from_results(results: Sequence[Dict[str, Any]]) -> set:
    """Aggregate pitch classes from a list of analysis results."""

    pcs: set[int] = set()
    for res in results:
        if isinstance(res, AnalysisResult):
            best_candidate = res.best.to_dict() if res.best else {}
        else:
            best_candidate = res.get("best") if isinstance(res, dict) else {}
            if isinstance(best_candidate, MatchCandidate):
                best_candidate = best_candidate.to_dict()
        labels = (best_candidate or {}).get("notes_oct") or []
        if not labels:
            labels = (best_candidate or {}).get("notes") or []
        for label in labels:
            if not label:
                continue
            note = label
            while note and note[-1].isdigit():
                note = note[:-1]
            if note in NOTE_VALUES:
                pcs.add(NOTE_VALUES[note])
    return pcs


# ---------------------- Formatting utilities ----------------------

def _coerce_candidate_dict(candidate: Any) -> Optional[Dict[str, Any]]:
    if candidate is None:
        return None
    if isinstance(candidate, MatchCandidate):
        return candidate.to_dict()
    if isinstance(candidate, dict):
        return candidate
    return None


def format_analysis_fr(result: Any) -> str:
    """Format an analysis result in direct French language."""

    if isinstance(result, AnalysisResult):
        payload = result.to_dict()
    elif isinstance(result, dict):
        payload = result
    else:
        # Treat a candidate as a minimal analysis result.
        candidate_dict = _coerce_candidate_dict(result)
        if not candidate_dict:
            raise TypeError("Unsupported result type for formatting")
        payload = {
            "best": candidate_dict,
            "alternatives": [candidate_dict],
            "error": None,
        }

    if payload.get("error"):
        return f"Analyse impossible : {payload['error']}"

    best = _coerce_candidate_dict(payload.get("best"))
    if not best:
        return "Aucune correspondance trouvée pour l'entrée fournie."

    lines = []
    lines.append(
        "Résultat : {line} — ⭐{stars}/5".format(
            line=format_for_syntakt(best),
            stars=best.get("stars", "?"),
        )
    )
    notes_oct = best.get("notes_oct") or []
    notes = best.get("notes") or []
    if notes_oct:
        lines.append("Notes jouées : " + " ".join(notes_oct))
    elif notes:
        lines.append("Notes jouées : " + " ".join(notes))
    explanation = (best.get("metrics") or {}).get("explanation")
    if explanation:
        lines.append("Explication : " + explanation)
    alternatives = payload.get("alternatives") or []
    alt_count = len(alternatives)
    if alt_count > 1:
        lines.append(f"Alternatives disponibles : {alt_count - 1} autres propositions classées.")
    return "\n".join(lines)


def format_for_syntakt(result: Any) -> str:
    """Return the canonical ``Root=... Preset=... BAL=...`` line."""

    if isinstance(result, MatchCandidate):
        return result.copy_line or _format_copy_line(
            result.sy_root,
            result.sy_type,
            result.bal,
            result.bal_motif,
        )
    if isinstance(result, AnalysisResult):
        return format_for_syntakt(result.best) if result.best else ""
    if isinstance(result, dict):
        if "sy_root" in result and "sy_type" in result and "bal" in result:
            motif = result.get("bal_motif") or result.get("bal_pattern") or ""
            bal_value = int(result.get("bal"))
            return _format_copy_line(result["sy_root"], result["sy_type"], bal_value, motif)
        best = result.get("best")
        if best:
            return format_for_syntakt(best)
        return ""
    return ""


def normalize_input(input_text: str) -> Dict[str, Any]:
    """Normalise free-form user input for downstream processing."""

    kind, payload = parse_free_input(input_text)
    data: Dict[str, Any] = {"kind": kind, "raw": input_text}
    if kind == "symbol":
        root, quality = payload
        pcs, rel, norm_root, root_pc = real_chord_pcs_and_order(root, quality)
        data.update(
            {
                "root": norm_root,
                "quality": quality,
                "pcs": pcs,
                "rel_order": rel,
                "root_pc": root_pc,
                "bass": norm_root,
                "inversion": 0,
            }
        )
    else:
        notes = list(payload)
        data["notes"] = notes
        data["bass"] = notes[0]
        rel = rel_from_note_list(notes)
        data["rel_order"] = rel
        data["inversion"] = TRIAD_INVERSION_MAP.get(tuple(rel))
    return data


def get_doc(section: str, key: Optional[str] = None) -> Any:
    """Return documentation entries from ``syntakt_documentation.json``."""

    doc = _load_doc_cache()
    sec = doc.get(section, {}) if isinstance(doc, dict) else {}
    if key is None:
        return sec
    if isinstance(sec, dict):
        return sec.get(key)
    return None


def get_version() -> str:
    """Return the current engine version string."""

    return ENGINE_VERSION
