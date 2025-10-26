"""Compatibility module forwarding to :mod:`sychord.core`."""
from __future__ import annotations

try:  # pragma: no cover
    from ..sychord.core import *  # type: ignore F401,F403
except Exception:  # pragma: no cover
    from sychord.core import *  # type: ignore F401,F403
