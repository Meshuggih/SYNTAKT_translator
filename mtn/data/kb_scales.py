# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple

# 36 échelles Syntakt (Appendix Keyboard Scales)
KB_SCALES: Dict[str, List[str]] = {
  "CHROMATIC": ["1","b2","2","#2","3","4","#4","5","#5","6","#6","7"],
  "IONIAN (MAJOR)": ["1","2","3","4","5","6","7"],
  "DORIAN": ["1","2","b3","4","5","6","b7"],
  "PHRYGIAN": ["1","b2","b3","4","5","b6","b7"],
  "LYDIAN": ["1","2","3","#4","5","6","7"],
  "MIXOLYDIAN": ["1","2","3","4","5","6","b7"],
  "AEOLIAN (MINOR)": ["1","2","b3","4","5","b6","b7"],
  "LOCRIAN": ["1","b2","b3","4","b5","b6","b7"],
  "PENTATONIC MINOR": ["1","b3","4","5","b7"],
  "PENTATONIC MAJOR": ["1","2","3","5","6"],
  "MELODIC MINOR": ["1","2","b3","4","5","6","7"],
  "HARMONIC MINOR": ["1","2","b3","4","5","b6","7"],
  "WHOLE TONE": ["1","2","3","#4","#5","b7"],
  "BLUES": ["1","b3","4","b5","5","b7"],
  "COMBO MINOR": ["1","2","b3","4","5","b6","b7","7"],  # hypothèse de travail
  "PERSIAN": ["1","b2","3","4","b5","b6","7"],
  "IWATO": ["1","b2","4","b5","b7"],
  "IN-SEN": ["1","b2","4","5","b7"],
  "HIRAJOSHI": ["1","2","b3","5","b6"],
  "PELOG": ["1","b2","b3","b4","5","b6"],  # approx. ET
  "PHRYGIAN DOMINANT": ["1","b2","3","4","5","b6","b7"],
  "WHOLE-HALF DIMINISHED": ["1","2","b3","4","b5","#5","6","7"],
  "HALF-WHOLE DIMINISHED": ["1","b2","#2","3","#4","5","6","b7"],
  "SPANISH": ["1","b2","b3","3","4","b5","b6","b7"],
  "MAJOR LOCRIAN": ["1","2","3","4","b5","b6","b7"],
  "SUPER LOCRIAN": ["1","b2","#2","3","b5","#5","b7"],
  "DORIAN b2": ["1","b2","b3","4","5","6","b7"],
  "LYDIAN AUGMENTED": ["1","2","3","#4","#5","6","7"],
  "LYDIAN DOMINANT": ["1","2","3","#4","5","6","b7"],
  "DOUBLE HARMONIC MAJOR": ["1","b2","3","4","5","b6","7"],
  "LYDIAN #2 #6": ["1","#2","3","#4","5","#6","7"],
  "ULTRAPHRYGIAN": ["1","b2","b3","b4","5","b6","bb7"],
  "HUNGARIAN MINOR": ["1","2","b3","#4","5","b6","7"],
  "ORIENTAL": ["1","b2","3","4","b5","6","b7"],
  "IONIAN #2 #5": ["1","#2","3","4","#5","6","7"],
  "LOCRIAN bb3 bb7": ["1","b2","bb3","4","b5","b6","bb7"],
}
DEGREE_TO_ST = {"1":0,"b2":1,"#2":3,"2":2,"bb3":2,"b3":3,"3":4,"b4":4,"4":5,"#4":6,"b5":6,"5":7,"#5":8,"b6":8,"6":9,"#6":10,"bb7":9,"b7":10,"7":11}


def scale_pcs(scale_name: str, root_pc: int) -> set:
    return { (root_pc + DEGREE_TO_ST[d]) % 12 for d in KB_SCALES[scale_name] }


def recommend_kb_scale(chord_pcs: set, *, policy: str = "safe") -> List[Tuple[str,int,float]]:
    """
    Retourne une liste triée (scale_name, root_pc, score).
    score = couverture (Jaccard) - pénalité de clash (selon la policy).
    """
    out = []
    for root_pc in range(12):
        for name in KB_SCALES:
            pcs = scale_pcs(name, root_pc)
            cover = len(pcs & chord_pcs) / len(pcs | chord_pcs) if (pcs or chord_pcs) else 1.0
            clash = 0.0
            if policy == "safe":
                accidental_count = sum(degree.count("b") + degree.count("#") for degree in KB_SCALES[name])
                clash = 0.005 * accidental_count
            out.append((name, root_pc, cover - clash))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:8]
