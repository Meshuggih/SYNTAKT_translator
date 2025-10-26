"""API publique du traducteur Syntakt."""
from SyntaktTranslatorV3 import Session, format_analysis_fr
from kb_scales import recommend_kb_scale

__version__ = "3.x"

__all__ = ["Session", "format_analysis_fr", "recommend_kb_scale", "__version__"]
