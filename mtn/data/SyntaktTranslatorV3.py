# -*- coding: utf-8 -*-
"""Syntakt SY CHORD translator with optional Pythonista UI glue."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List

try:  # Optional Pythonista-only modules
    import ui  # type: ignore
    import clipboard  # type: ignore
    import console  # type: ignore
    from objc_util import ObjCInstance  # type: ignore
except Exception:  # pragma: no cover - absent on headless environments
    ui = None  # type: ignore
    clipboard = None  # type: ignore
    console = None  # type: ignore
    ObjCInstance = None  # type: ignore

from .syntakt_core import (
    Session,
    aggregated_chord_pcs_from_results,
    format_analysis_fr,
    recommend_kb_scale,
)

__all__ = [
    "Session",
    "format_analysis_fr",
    "analyze_text",
    "get_session",
    "has_pythonista_ui",
    "advise_keyboard_scale",
]


_session: Optional[Session] = None


def get_session() -> Session:
    """Return a module-level session reusing the Syntakt library."""
    global _session
    if _session is None:
        _session = Session()
    return _session


def analyze_text(input_text: str, **options: Any) -> Dict[str, Any]:
    """Convenience helper returning ``Session().analyze`` with shared cache."""
    return get_session().analyze(input_text, **options)


def advise_keyboard_scale(symbols: Sequence[str], policy: str = "safe") -> List[Dict[str, Any]]:
    """Return top keyboard scale recommendations for a chord progression."""

    session = get_session()
    analyses = [session.analyze(sym) for sym in symbols]
    pcs = aggregated_chord_pcs_from_results(analyses)
    top = recommend_kb_scale(pcs, policy=policy)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return [
        {"kb_scale": scale, "root": names[root_pc], "score": round(score, 3)}
        for (scale, root_pc, score) in top
    ]


def has_pythonista_ui() -> bool:
    """Expose whether Pythonista specific modules are available."""
    return ui is not None


if ui is not None:

    class TranslatorView(ui.View):  # pragma: no cover - exercised on device
        """Minimal UI wrapper delegating logic to :class:`Session`."""

        def __init__(self) -> None:
            super().__init__()
            self.name = "Syntakt SY CHORD"
            self.background_color = "white"
            self._text = ui.TextView(frame=(10, 10, self.width - 20, 120), flex="WH")
            self._text.autocorrection_type = False
            self._text.font = ("<System>", 16)
            self.add_subview(self._text)
            self._button = ui.Button(title="Analyser", frame=(10, 140, self.width - 20, 40), flex="WT")
            self._button.action = self._on_analyze
            self.add_subview(self._button)
            self._output = ui.TextView(frame=(10, 190, self.width - 20, self.height - 200), flex="WH")
            self._output.editable = False
            self._output.font = ("<System>", 14)
            self.add_subview(self._output)

        def layout(self) -> None:
            self._text.width = self.width - 20
            self._button.width = self.width - 20
            self._output.width = self.width - 20
            self._output.height = self.height - 200

        def _on_analyze(self, sender: Any) -> None:
            session = get_session()
            result = session.analyze(self._text.text or "")
            self._output.text = format_analysis_fr(result)
            if clipboard is not None and result.get("copy_lines"):
                clipboard.set(result["copy_lines"][0])
            if console is not None:
                console.hud_alert("Analyse terminée", "success", 0.7)

    def present_ui() -> None:  # pragma: no cover
        view = TranslatorView()
        view.present("sheet")

else:

    class TranslatorView:  # pragma: no cover - placeholder for type checkers
        def __init__(self) -> None:
            raise RuntimeError("L'interface Pythonista n'est pas disponible dans ce contexte.")

    def present_ui() -> None:  # pragma: no cover
        raise RuntimeError("Interface Pythonista indisponible.")


__all__.append("present_ui")

