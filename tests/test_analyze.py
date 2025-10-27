import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mtn" / "data"))

from SyntaktTranslatorV3 import analyze  # noqa: E402


def pick_best(payload):
    return payload["best"]


def test_am_basic():
    result = analyze("Am")
    best = pick_best(result)
    assert best["root"] == "A"
    assert best["preset"] in ("minor", "m")
    assert isinstance(best.get("voicing"), str) and best["voicing"]
    assert result["copy_line"].startswith("Root=A  Preset=")


def test_fmaj7():
    result = analyze("FMaj7")
    assert result["best"]["preset"].lower() == "maj7"


def test_g7():
    result = analyze("G7")
    assert result["best"]["preset"] == "7"


def test_bm7b5():
    result = analyze("Bm7b5")
    assert result["best"]["preset"].lower() == "m7b5"


def test_notes_second_inversion_bal106():
    result = analyze("E A C")
    best = result["best"]
    assert best["root"] in ("A", "A")
    assert best["preset"].startswith("m")
    assert best["bal"] == 106
    voicing = best["voicing"]
    assert "C4" in voicing and "E4" in voicing and "A4" in voicing


def test_dbadd9_enharmonic():
    result = analyze("Dbadd9")
    best = result["best"]
    assert best["preset"] == "add9"
    assert best["root"] in ("Db", "C#")


def test_fsus2():
    result = analyze("Fsus2")
    assert result["best"]["preset"] == "sus2"


def test_em6_symbol_now_parses():
    result = analyze("Em6")
    best = result["best"]
    assert best["root"] == "E"
    assert best["preset"] in ("m6", "minor6")
    assert "C#" in " ".join(best["notes"])


def test_strict_option_respects_order():
    result = analyze("E A C", strict=True)
    assert result["best"] is not None
    assert isinstance(result["alternatives"], list)
