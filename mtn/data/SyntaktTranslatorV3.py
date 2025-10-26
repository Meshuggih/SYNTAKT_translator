"""Compatibility module forwarding to :mod:`sychord.translator`."""
from __future__ import annotations

try:  # pragma: no cover
    from ..sychord.translator import *  # type: ignore F401,F403
except Exception:  # pragma: no cover
    from sychord.translator import *  # type: ignore F401,F403
