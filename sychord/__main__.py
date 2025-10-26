"""Command line interface for the Syntakt SY CHORD translator."""
from __future__ import annotations

import argparse
import json
from typing import Any

from . import Session, format_analysis_fr, format_for_syntakt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Syntakt SY CHORD analyzer")
    parser.add_argument("input", help="Chord symbol or list of notes to analyse")
    parser.add_argument("--topk", type=int, default=6, help="Number of alternatives to return")
    parser.add_argument(
        "--anchor-octave",
        type=int,
        default=3,
        help="Octave anchor for generated voicings",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode – require matching order for note lists",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the full JSON payload instead of the human summary",
    )
    args = parser.parse_args(argv)

    session = Session()
    result = session.analyze(
        args.input,
        topk=args.topk,
        anchor_octave=args.anchor_octave,
        strict=args.strict,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_analysis_fr(result))
        print()
        print(format_for_syntakt(result))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
