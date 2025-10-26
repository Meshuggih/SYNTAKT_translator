"""Compat shim re-exporting the modern :mod:`sychord` package."""
from __future__ import annotations

try:  # pragma: no cover - compatibility path
    from ..sychord import Session, format_analysis_fr
    from ..sychord.core import recommend_kb_scale
except Exception:  # pragma: no cover
    from sychord import Session, format_analysis_fr  # type: ignore
    from sychord.core import recommend_kb_scale  # type: ignore

__all__ = ["Session", "format_analysis_fr", "recommend_kb_scale"]
