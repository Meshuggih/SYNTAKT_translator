import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from mtn.data.SyntaktTranslatorV3 import advise_keyboard_scale
from mtn.data.syntakt_core import Session, format_analysis_fr, voices_with_octaves


def test_analyze_symbol_returns_best_candidate():
    session = Session()
    result = session.analyze("Am", topk=3)
    best = result["best"]
    assert best is not None
    assert best["sy_root"] == "A"
    assert best["sy_type"] == "minor"
    assert result["copy_lines"][0].startswith("Root=A Preset=minor")
    assert best["metrics"]["explanation"] == "exact"


def test_analyze_notes_supports_octave_anchor():
    session = Session()
    result = session.analyze("E A C", anchor_octave=2, topk=2)
    best = result["best"]
    assert best is not None
    assert best["notes_oct"][0].startswith("E2")
    assert len(result["alternatives"]) <= 2


def test_formatting_in_french():
    session = Session()
    result = session.analyze("Fm7")
    formatted = format_analysis_fr(result)
    assert "Résultat :" in formatted
    assert "Notes jouées" in formatted


def test_analyze_reports_errors():
    session = Session()
    result = session.analyze("???")
    assert result["best"] is None
    assert result["error"]
    assert "Entrée non reconnue" in result["error"]


def test_madd9_places_ninth_above_octave():
    session = Session()
    cand = next(
        c
        for c in session.library
        if c["sy_root"] == "C" and c["sy_type"] == "madd9" and c["bal_pattern"] == "••••"
    )
    voices = voices_with_octaves(cand, anchor_octave=3)
    assert voices[0]["note"] == "C"
    assert voices[-1]["note"] == "D"
    assert voices[-1]["octave"] >= voices[0]["octave"] + 1


def test_dominant_preset_uses_m7_label():
    session = Session()
    result = session.analyze("G7")
    best = result["best"]
    assert best is not None
    assert best["sy_type"] == "M7"


def test_keyboard_scale_advisor_suggests_ionian_major():
    progression = ["Gmaj7", "D", "Em7", "Cmaj7"]
    recommendations = advise_keyboard_scale(progression)
    top_three = recommendations[:3]
    assert any(
        rec["kb_scale"] == "IONIAN (MAJOR)" and rec["root"] == "G"
        for rec in top_three
    )
