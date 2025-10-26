import importlib

import pytest

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'mtn' / 'data'))


def test_imports_are_flat():
    mod = importlib.import_module("SyntaktTranslatorV3")
    kb = importlib.import_module("kb_scales")
    assert hasattr(mod, "Session")
    assert hasattr(mod, "format_analysis_fr")
    assert hasattr(kb, "recommend_kb_scale")


def test_session_defaults_and_error_message():
    from SyntaktTranslatorV3 import Session

    session = Session()
    result = session.analyze("???")
    assert result["options"]["strict"] is False
    assert result["options"]["pref_bass_root"] is True
    assert "Entrée" in result["error"]


def test_symbol_and_notes_analysis():
    from SyntaktTranslatorV3 import Session

    session = Session()
    result_symbol = session.analyze("G7")
    best = result_symbol["best"]
    assert best is not None
    assert best["sy_root"] == "G"
    assert best["stars"] >= 3

    result_notes = session.analyze("E A C", anchor_octave=2)
    best_notes = result_notes["best"]
    assert best_notes is not None
    assert best_notes["bal"] == 106
    assert best_notes["notes_oct"][0].startswith("E3")


def test_triad_second_inversion_prefers_bal_106():
    from SyntaktTranslatorV3 import Session

    session = Session()
    result = session.analyze("E A C", topk=4)
    best = result["best"]
    assert best is not None
    assert best["bal"] == 106
    assert best["metrics"]["bal_preference_rank"] == 0
    assert any(alt["bal"] == 74 for alt in result["alternatives"][1:])


def test_tetrad_coverage_and_scoring():
    from SyntaktTranslatorV3 import Session

    session = Session()
    for symbol in ["G7", "Bm7b5", "CMaj7", "Am7", "Eadd9"]:
        result = session.analyze(symbol)
        best = result["best"]
        assert best is not None, symbol
        assert best["stars"] >= 3
        assert best["metrics"]["missing"] == 0
        assert "Root=" in best["copy_line"]


def test_format_analysis_fr_mentions_alternatives():
    from SyntaktTranslatorV3 import Session, format_analysis_fr

    session = Session()
    result = session.analyze("Am", topk=5)
    formatted = format_analysis_fr(result)
    assert "Résultat" in formatted
    assert "Alternatives" in formatted


def test_keyboard_scale_recommendation():
    from kb_scales import recommend_kb_scale

    recommendations = recommend_kb_scale({7, 11, 2, 5})
    assert any(name == "IONIAN (MAJOR)" for name, _, _ in recommendations)


@pytest.mark.parametrize(
    "entry,expected_bass",
    [
        ("C E G", "C"),
        ("B D F", "B"),
    ],
)
def test_normalized_voicing_keeps_order(entry, expected_bass):
    from SyntaktTranslatorV3 import Session

    session = Session(anchor_octave=1)
    result = session.analyze(entry)
    best = result["best"]
    assert best is not None
    assert best["notes_oct"][0].startswith(expected_bass)
