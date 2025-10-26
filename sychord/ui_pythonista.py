"""Optional Pythonista UI glue for the Syntakt translator."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

try:  # pragma: no cover - executed only on Pythonista
    import ui  # type: ignore
    import clipboard  # type: ignore
    import console  # type: ignore

    HAS_PYTHONISTA_UI = True
except Exception:  # pragma: no cover - normal on headless environments
    ui = None  # type: ignore
    clipboard = None  # type: ignore
    console = None  # type: ignore
    HAS_PYTHONISTA_UI = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import Session


def present_ui(*, session_factory: Callable[[], "Session"]) -> None:
    """Display the Pythonista UI when the environment supports it."""

    if not HAS_PYTHONISTA_UI:
        raise RuntimeError("Interface Pythonista indisponible dans ce contexte.")

    class TranslatorView(ui.View):  # type: ignore[misc]
        """Minimal UI wrapper delegating logic to :class:`Session`."""

        def __init__(self, factory: Callable[[], "Session"]) -> None:
            super().__init__()
            self._factory = factory
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
            session = self._factory()
            result = session.analyze(self._text.text or "")
            from .core import format_analysis_fr  # local import to avoid cycle

            self._output.text = format_analysis_fr(result)
            if clipboard is not None and result.get("copy_lines"):
                clipboard.set(result["copy_lines"][0])
            if console is not None:
                console.hud_alert("Analyse terminée", "success", 0.7)

    view = TranslatorView(session_factory)
    view.present("sheet")


__all__ = ["HAS_PYTHONISTA_UI", "present_ui"]
