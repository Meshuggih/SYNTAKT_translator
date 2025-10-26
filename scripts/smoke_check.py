#!/usr/bin/env python3
"""Vérification rapide de l'API Syntakt."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

EXPECTED_FILES = {
    "SyntaktTranslatorV3.py",
    "syntakt_core.py",
    "kb_scales.py",
    "Syntakt.csv",
    "syntakt_documentation.json",
    "__init__.py",
}


def assert_flat_dataset() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "mtn" / "data"
    if not data_dir.exists():
        raise SystemExit("dossier mtn/data introuvable")
    found = {path.name for path in data_dir.iterdir() if not path.name.startswith('.')} \
        - {"__pycache__"}
    if found != EXPECTED_FILES:
        missing = EXPECTED_FILES - found
        extra = found - EXPECTED_FILES
        raise SystemExit(
            f"Contenu inattendu dans mtn/data. Manquants: {sorted(missing)}, Supplémentaires: {sorted(extra)}"
        )
    if len(found) > 20:
        raise SystemExit("Trop de fichiers dans mtn/data")


def main() -> None:
    assert_flat_dataset()
    data_dir = Path(__file__).resolve().parents[1] / "mtn" / "data"
    sys.path.insert(0, str(data_dir))
    syntakt = importlib.import_module("SyntaktTranslatorV3")
    kb = importlib.import_module("kb_scales")
    session = syntakt.Session()
    result = session.analyze("G7")
    line = result["copy_lines"][0] if result["copy_lines"] else "<aucune>"
    print("Analyse G7 :", line)
    recs = kb.recommend_kb_scale(set(result["chord_pcs"]))
    print("Top scale :", recs[0])


if __name__ == "__main__":
    main()
