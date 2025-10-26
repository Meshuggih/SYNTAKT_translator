import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from mtn.data.syntakt_core import Session, format_analysis_fr


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
