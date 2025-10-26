"""Compatibility module forwarding to :mod:`sychord.core`."""
from __future__ import annotations

try:  # pragma: no cover
    from ..sychord.core import recommend_kb_scale  # type: ignore
except Exception:  # pragma: no cover
    from sychord.core import recommend_kb_scale  # type: ignore

__all__ = ["recommend_kb_scale"]
