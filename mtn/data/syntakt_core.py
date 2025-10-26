"""Core logic for Syntakt chord analysis, independent from Pythonista APIs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re

from .kb_scales import recommend_kb_scale

__all__ = [
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
    "Session",
    "format_analysis_fr",
    "aggregated_chord_pcs_from_results",
    "recommend_kb_scale",
]

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
    "major": [0, 4, 7],
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
    "m9no5": [0, 2, 3, 10],
    "M9no5": [0, 4, 10, 14],
    "Madd9b5": [0, 2, 4, 6],
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
) -> List[Dict[str, Any]]:
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
    out = []
    for c in cands[:64]:
        metrics = rate_symbol_candidate(c, rr_norm, quality, rpcs, rrel)
        stars = stars_from_symbol_metrics(metrics, pref_bass, pref_root)
        v = voices_with_octaves(c, anchor_octave=anchor_octave)
        c2 = dict(c)
        c2["notes_oct"] = [d["label"] for d in v]
        c2["bal_motif"] = c.get("bal_pattern", "")
        c2["metrics"] = metrics
        c2["stars"] = stars
        out.append(c2)
    out.sort(key=lambda cand: (-cand["stars"], cand["metrics"]))
    return out[:topk]


def list_matches_notes(
    user_notes: Sequence[str],
    sy_lib: Sequence[Dict[str, Any]],
    anchor_octave: int,
    topk: int,
    strict: bool,
    pref_bass: bool,
) -> List[Dict[str, Any]]:
    user_set = sorted({NOTE_VALUES[n] for n in user_notes})
    user_rel = rel_from_note_list(user_notes)
    cands = [c for c in sy_lib if set(c["pcs_set"]) == set(user_set)]
    if strict:
        cands = [c for c in cands if c["rel_order"] == user_rel]

    def _key(c: Dict[str, Any]) -> Tuple[int, int, int, int]:
        inv = 0 if c["rel_order"] == user_rel else 1
        bass_pc = NOTE_VALUES[user_notes[0]]
        bass_match = 0 if c["root_pc"] == bass_pc else (2 if pref_bass else 1)
        bal_pref = triad_bal_pref_rank(c)
        return (inv, bass_match, bal_pref, -c.get("nvoices", 4))

    cands = sorted(cands, key=_key)
    out = []
    for c in cands[:64]:
        metrics = rate_notes_candidate(c, user_notes)
        stars = stars_from_notes_metrics(metrics, pref_bass=pref_bass)
        v = voices_with_octaves(c, anchor_octave=anchor_octave)
        c2 = dict(c)
        c2["notes_oct"] = [d["label"] for d in v]
        c2["bal_motif"] = c.get("bal_pattern", "")
        c2["metrics"] = metrics
        c2["stars"] = stars
        out.append(c2)
    out.sort(key=lambda cand: (-cand["stars"], cand["metrics"]))
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


@dataclass
class AnalysisResult:
    query: Dict[str, Any]
    best: Optional[Dict[str, Any]]
    alternatives: List[Dict[str, Any]]
    copy_lines: List[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "best": self.best,
            "alternatives": self.alternatives,
            "copy_lines": self.copy_lines,
            "error": self.error,
        }


class Session:
    """Headless analysis session for Syntakt chord matching."""

    def __init__(
        self,
        library: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        octify_extensions: bool = True,
    ) -> None:
        self.library = (
            list(library)
            if library is not None
            else build_syntakt_library(octify_extensions=octify_extensions)
        )

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
        """Analyse ``input_text`` and return a structured dictionary.

        Keys of the returned dictionary:

        * ``query`` – metadata about the user request (type, payload, options).
        * ``best`` – the highest ranked candidate with ``notes``, ``notes_oct``,
          ``stars`` and ``metrics`` fields. ``metrics`` follows the structure of
          :func:`_metrics_to_dict` and the accompanying ``metrics_raw`` keeps the
          original ranking tuple.
        * ``alternatives`` – list of enriched candidates sorted by stars.
        * ``copy_lines`` – ready-to-copy strings (``Root=... Preset=... BAL=...``).
        * ``error`` – error message when parsing fails (otherwise ``None``).
        """

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
            return AnalysisResult(query=query_info, best=None, alternatives=[], copy_lines=[], error=str(exc)).to_dict()

        enriched: List[Dict[str, Any]] = []
        for cand in alts:
            c = dict(cand)
            raw_metrics = cand["metrics"]
            c["metrics"] = _metrics_to_dict(raw_metrics)
            c["metrics_raw"] = list(raw_metrics)
            c.setdefault("notes", list(cand.get("notes", [])))
            c.setdefault("notes_oct", list(cand.get("notes_oct", [])))
            enriched.append(c)

        best = enriched[0] if enriched else None
        copy_lines = [
            f"Root={c['sy_root']} Preset={c['sy_type']} BAL={c['bal']} ({c.get('bal_motif', '')})"
            for c in enriched
        ]
        return AnalysisResult(query=query_info, best=best, alternatives=enriched, copy_lines=copy_lines).to_dict()


# ---------------------- Keyboard advisor helpers ----------------------

def aggregated_chord_pcs_from_results(results: Sequence[Dict[str, Any]]) -> set:
    """Aggregate pitch classes from a list of analysis results."""

    pcs: set[int] = set()
    for res in results:
        best = res.get("best") or {}
        labels = best.get("notes_oct") or []
        if not labels:
            labels = best.get("notes") or []
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

def format_analysis_fr(result: Dict[str, Any]) -> str:
    """Format an analysis result in direct French language."""
    if result.get("error"):
        return f"Analyse impossible : {result['error']}"
    best = result.get("best")
    if not best:
        return "Aucune correspondance trouvée pour l'entrée fournie."
    lines = []
    lines.append(
        "Résultat : Root={root} Preset={preset} BAL={bal} {motif} — ⭐{stars}/5".format(
            root=best["sy_root"],
            preset=best["sy_type"],
            bal=best["bal"],
            motif=best.get("bal_motif", ""),
            stars=best.get("stars", "?"),
        )
    )
    if best.get("notes_oct"):
        lines.append("Notes jouées : " + " ".join(best["notes_oct"]))
    elif best.get("notes"):
        lines.append("Notes jouées : " + " ".join(best["notes"]))
    explanation = best.get("metrics", {}).get("explanation")
    if explanation:
        lines.append("Explication : " + explanation)
    alt_count = len(result.get("alternatives", []))
    if alt_count > 1:
        lines.append(f"Alternatives disponibles : {alt_count - 1} autres propositions classées.")
    return "\n".join(lines)
