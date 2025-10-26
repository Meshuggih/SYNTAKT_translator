# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Syntakt SY CHORD — Convertisseur (UI Pythonista 3 + Ranking + Octaves)
======================================================================

- Entrée flexible : accord (ex. "Am", "FMaj7") OU notes graves→aigu (ex. "E A C", "EAC").
- Sortie : Root / Preset / BAL (+ motif), notes **avec octaves** (ex. G3–C4–E4), score + **étoiles**.
- Classement par fidélité (5★ à 1★), mode Strict (sans inversion), préférences de voicing.
- Export JSON/YAML/MD enrichi (étoiles + notes_oct), logs SQLite.
- UI ajustée safe-area (iPhone 14 Pro Max), pickers, etc.

Important:
- BAL 106 = '•*••' ; BAL 74 = '••..' (logique d'inversion respectée).
- Types d’accords complétés : **Fourth** et **Fifth** (dyades), en plus de tous les autres.
"""

from typing import Dict, List, Tuple, Optional, Any
import ui, clipboard, console
import os, json, sqlite3, datetime, re
from objc_util import ObjCInstance

# ---------------------- Couleurs / Constantes UI ----------------------
IOS_BLUE = (0.0, 0.478, 1.0)
IOS_GREEN = (0.2, 0.8, 0.2)
GRAY = (0.45, 0.45, 0.47)
BORDER = (0.82, 0.82, 0.85)
FG = (0.0, 0.0, 0.0)
BG = (1.0, 1.0, 1.0)

DOCS = os.path.expanduser('~/Documents')
EXPORT_DIR = os.path.join(DOCS, 'sy_chord_exports')
LOG_DB = os.path.join(DOCS, 'sy_chord_logs.db')
README = os.path.join(DOCS, 'SY_CHORD_README.md')
RECAPI_MD = os.path.join(DOCS, 'sy_chord_session_log.md')

if not os.path.isdir(EXPORT_DIR):
    try:
        os.makedirs(EXPORT_DIR)
    except Exception:
        pass

# ---------------------- Safe Area utils ----------------------
def get_safe_insets(py_view: ui.View) -> Tuple[float, float, float, float]:
    """Retourne (top, right, bottom, left) des safe areas, fallback si indispo."""
    try:
        v = ObjCInstance(py_view)
        insets = v.safeAreaInsets()
        top = float(insets.top)
        left = float(insets.left)
        bottom = float(insets.bottom)
        right = float(insets.right)
        return top, right, bottom, left
    except Exception:
        # iPhone 14 Pro Max: top≈59, bottom≈34
        return 59.0, 0.0, 34.0, 0.0

# ---------------------- Notes & Normalisation ----------------------
NOTE_VALUES: Dict[str, int] = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4,
    'F': 5, 'E#': 5,
    'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11, 'B#': 0,
}

# affichage préférant dièses
VALUES_TO_NOTES: Dict[int, str] = {}
for k, v in NOTE_VALUES.items():
    if v not in VALUES_TO_NOTES:
        VALUES_TO_NOTES[v] = k
for k, v in [('C',0),('C#',1),('D',2),('D#',3),('E',4),('F',5),
             ('F#',6),('G',7),('G#',8),('A',9),('A#',10),('B',11)]:
    VALUES_TO_NOTES[v] = k

ALL_ROOTS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def normalize_note(name: str) -> str:
    name = (name or '').strip()
    if name in NOTE_VALUES:
        return VALUES_TO_NOTES[NOTE_VALUES[name]]
    u = name.upper().replace('BEMOL','B').replace('DIESE','#')
    if u in NOTE_VALUES:
        return VALUES_TO_NOTES[NOTE_VALUES[u]]
    raise ValueError(f"Note invalide: {name}")

def midi_from_note_oct(name: str, octv: int) -> int:
    """C-1 = 0 → C4 = 60."""
    return 12*(octv+1) + NOTE_VALUES[name]

def note_oct_from_midi(m: int) -> Tuple[str,int]:
    return VALUES_TO_NOTES[m%12], (m//12)-1

# ---------------------- Accords "réels" ----------------------
REAL_TRIADS: Dict[str, List[int]] = {
    'major': [0, 4, 7],
    'minor': [0, 3, 7],
    'diminished': [0, 3, 6],
    'augmented': [0, 4, 8],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
}
REAL_TETRADS: Dict[str, List[int]] = {
    'major7': [0, 4, 7, 11],
    'minor7': [0, 3, 7, 10],
    'dominant7': [0, 4, 7, 10],  # "7"
    'minorMajor7': [0, 3, 7, 11],
    'diminished7': [0, 3, 6, 9],
    'halfDim7': [0, 3, 6, 10],    # m7b5
    'major6': [0, 4, 7, 9],
    'minor6': [0, 3, 7, 9],
    'augMajor7': [0, 4, 8, 11],
    'add9_major': [0, 4, 7, 14],
    'add9_minor': [0, 3, 7, 14],
}
REAL_SYNONYMS: Dict[str, str] = {
    'maj': 'major', 'min': 'minor', 'dim': 'diminished', 'aug': 'augmented',
    'M7': 'major7', 'maj7': 'major7',  # M7 -> major7
    'm7': 'minor7', 'min7': 'minor7',
    '7': 'dominant7', 'dom7': 'dominant7',
    'mMaj7': 'minorMajor7', 'mmaj7': 'minorMajor7', 'minMaj7': 'minorMajor7',
    'dim7': 'diminished7',
    'm7b5': 'halfDim7', 'ø7': 'halfDim7',
    'M6': 'major6', '6': 'major6',
    'm6': 'minor6',
    'augM7': 'augMajor7',
    'add9': 'add9_major', 'madd9': 'add9_minor',
}
def real_chord_offsets(quality: str) -> List[int]:
    q = (quality or '').strip()
    if q in REAL_SYNONYMS:
        q = REAL_SYNONYMS[q]
    if q in REAL_TRIADS:
        return REAL_TRIADS[q]
    if q in REAL_TETRADS:
        return REAL_TETRADS[q]
    raise ValueError(f"Qualité d'accord inconnue: {quality}")

# ---------------------- SY CHORD presets (intervalles) ----------------------
CHORD_INTERVALS: Dict[str, List[int]] = {
    # --- Unisons ---
    'unison1': [0],
    'unison2': [0, 0],
    'unison3': [0, 0, 0],
    'unison4': [0, 0, 0, 0],

    # --- Dyades (Appendix complet) ---
    'Fourth': [0, 5],   # quarte juste
    'Fifth':  [0, 7],   # quinte juste

    # --- Triades ---
    'minor': [0, 3, 7],
    'major': [0, 4, 7],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],

    # --- Tétrades ---
    'm7': [0, 3, 7, 10],
    '7':  [0, 4, 7, 10],      # dominante
    'Maj7':[0, 4, 7, 11],     # majeur 7 (alias M7 via REAL_SYNONYMS)
    'mMaj7':[0, 3, 7, 11],
    '7sus4':[0, 5, 7, 10],
    'dim7':[0, 3, 6, 9],

    # --- Variantes utiles ---
    'madd9':[0, 2, 3, 7],
    'Madd9':[0, 2, 4, 7],
    'm6':[0, 3, 7, 9],
    'M6':[0, 4, 7, 9],
    'mb5':[0, 3, 6],
    'Mb5':[0, 4, 6],
    'm7b5':[0, 3, 6, 10],
    'M7b5':[0, 4, 6, 10],
    'M#5':[0, 4, 8],
    'm7#5':[0, 3, 8, 10],
    'M7#5':[0, 4, 8, 10],
    'mb6':[0, 3, 8],
    'm9no5':[0, 2, 3, 10],
    'M9no5':[0, 2, 4, 10],
    'Madd9b5':[0, 2, 4, 6],
    'Maj7b5':[0, 4, 6, 11],
    'M7b9no5':[0, 1, 4, 10],
    'sus4#5b9':[0, 1, 5, 8],
    'sus4add#5':[0, 5, 7, 8],
    'Maddb5':[0, 4, 6, 7],
    'M6add4no5':[0, 4, 5, 9],
    'Maj7/6no5':[0, 4, 9],
    'Maj9no5':[0, 2, 4, 11],
}

# ---------------------- BAL → Motifs ----------------------
# Légende : • = 0 ; . = -12 ; * = +12 ; ~ = muet ; 'root' = voicing compact
BAL_PATTERNS: Dict[int, str] = {
    0: '•~~~', 10: '••~~', 20: '•••~', 30: '••••', 32: 'root',
    37:'•~••', 42: '•.••', 47: '•.~•', 52: '•..•', 57: '•..~',
    62:'•...',   69: '•~..', 74: '••..', 79: '••~.', 84: '•••.',
    91:'•••~', 96: '••••', 101:'•~••',106:'•*••', 111:'•*~•',
    116:'•**•',121:'•**~',127:'•***',
}

def _pattern_to_oct(pattern: str, nvoices: int) -> List[Optional[int]]:
    """
    Convertit un motif BAL (• . * ~ ... root) en décalages d'octave pour nvoices.
    0 pour octave centrale, -12 pour '.', +12 pour '*', None pour '~'.
    Cas spéciaux: 'root'→'••••' ; '...' ou '...' → remplacés par '..'.
    """
    if pattern == 'root':
        pattern = '••••'
    pat = pattern.replace('...', '..').replace('...', '..')
    pat = (pat + '~~~~')[:max(1, min(4, nvoices))]
    out: List[Optional[int]] = []
    for ch in pat:
        if ch == '•': out.append(0)
        elif ch == '.': out.append(-12)
        elif ch == '*': out.append(+12)
        elif ch == '~': out.append(None)
        else: out.append(None)
    while len(out) < nvoices:
        out.append(None)
    return out[:nvoices]

# ---------------------- Librairie Syntakt ----------------------
def build_syntakt_library() -> List[Dict[str, Any]]:
    lib: List[Dict[str, Any]] = []
    for sy_root in ALL_ROOTS:
        rv = NOTE_VALUES[sy_root]
        for sy_type, offsets in CHORD_INTERVALS.items():
            nvoices = min(4, len(offsets))
            base_offsets = offsets[:nvoices]
            for bal, pat in BAL_PATTERNS.items():
                octs = _pattern_to_oct(pat, nvoices)
                abs_semi: List[int] = []
                for i, off in enumerate(base_offsets):
                    if i >= len(octs) or octs[i] is None:
                        continue
                    abs_semi.append(rv + off + octs[i])
                if not abs_semi:
                    continue
                abs_semi.sort()
                pcs = sorted({s % 12 for s in abs_semi})
                base = abs_semi[0]
                rel = [(s - base) % 12 for s in abs_semi]
                notes = [VALUES_TO_NOTES[s % 12] for s in abs_semi]
                lib.append({
                    'sy_root': sy_root,
                    'sy_type': sy_type,
                    'bal': bal,
                    'bal_pattern': pat,
                    'intervals': base_offsets,
                    'octs': octs,
                    'notes': notes,            # sans octave
                    'pcs_set': pcs,
                    'rel_order': rel,
                    'root_pc': base % 12,      # basse (classe de pitch)
                    'nvoices': len(notes),
                })
    return lib

# ---------------------- Préférences renversements (triades) -------
# Triade majeure: root=[0,4,7], 1er inv=[0,3,8], 2e inv=[0,5,9]
# Triade mineure: root=[0,3,7], 1er inv=[0,4,9], 2e inv=[0,5,8]
PREF_BAL_BY_TRIAD_REL = {
    # Majeur
    (0, 4, 7): [20, 30, 91],          # fondamentale → compact
    (0, 3, 8): [79, 30, 20],          # 1er renversement → ••~.
    (0, 5, 9): [106, 74, 30, 20],     # 2e renversement → **106 (•*••) d'abord**
    # Mineur
    (0, 3, 7): [20, 30, 91],
    (0, 4, 9): [79, 30, 20],
    (0, 5, 8): [106, 74, 30, 20],
}
def triad_bal_pref_rank(c: dict) -> int:
    """Rang de préférence BAL pour triade (0=meilleur), sinon 9."""
    if c.get('nvoices') != 3:
        return 9
    key = tuple(c.get('rel_order') or [])
    prefs = PREF_BAL_BY_TRIAD_REL.get(key)
    if not prefs:
        return 9
    try:
        return prefs.index(c['bal'])
    except ValueError:
        return 8  # triade, mais BAL non listé

# ---------------------- OCTAVES absolues ----------------------
def voices_with_octaves(cand: Dict, anchor_octave:int=3) -> List[Dict[str,Any]]:
    """
    Calcule les voix absolues (note+octave+MIDI) en ancrant la fondamentale Syntakt
    (sy_root) à l’octave 'anchor_octave'. Applique intervalles + motif d’octave (BAL),
    puis tri grave→aigu.
    """
    root = cand['sy_root']; intervals = cand['intervals']; octs = cand['octs']
    n = min(len(intervals), 4); base_midi = midi_from_note_oct(root, anchor_octave)
    mids = []
    for i in range(n):
        if i >= len(octs) or octs[i] is None: continue
        mids.append(base_midi + intervals[i] + octs[i])
    mids.sort()
    out=[]
    for m in mids:
        nname, octv = note_oct_from_midi(m)
        out.append({'note': nname, 'octave': octv, 'midi': m, 'label': f"{nname}{octv}"})
    return out

# ---------------------- Parsing utilisateur ----------------------
NOTE_TOKEN = r'[A-Ga-g](?:#|b)?'
def parse_free_input(user_input:str) -> Tuple[str, Any]:
    """
    Retourne ('notes', [notes]) si l’entrée ressemble à une suite de notes graves→aigu,
    sinon ('symbol', (root, quality)).
    """
    s = (user_input or '').strip()
    if not s:
        raise ValueError("Entrée vide.")
    # Essai "liste de notes" : "EAC", "E A C", "E-A-C", "E,A,C"
    cleaned = s.replace('-', ' ').replace(',', ' ').replace(';', ' ')
    tokens = re.findall(NOTE_TOKEN, cleaned)
    if tokens:
        # Vérif : si la concaténation des tokens recouvre bien l'input sans autres symboles
        cat = ''.join(t.upper() for t in tokens)
        only_notes = re.sub(r'[^A-Ga-g#b]', '', s).upper()
        if cat == only_notes and len(tokens) >= 2:
            return ('notes', [normalize_note(t) for t in tokens])
    # Sinon, essaye "accord" : racine + qualité
    m = re.match(r'^\s*('+NOTE_TOKEN+r')\s*(.*)$', s)
    if m:
        root = normalize_note(m.group(1))
        qual = (m.group(2) or 'major').strip()
        # normalisations usuelles
        qual = qual.replace('minor','m').replace('major','M').replace('min','m').replace('maj','M')
        if qual in REAL_SYNONYMS:
            qual = REAL_SYNONYMS[qual]
        if qual in REAL_TRIADS or qual in REAL_TETRADS:
            return ('symbol', (root, qual))
    raise ValueError("Entrée non reconnue. Donne un nom d’accord (ex: Am, FMaj7) OU des notes graves→aigu (ex: E A C).")

# ---------------------- Fidélité / Scores & Étoiles ----------------------
def jaccard(a:set,b:set)->float:
    return len(a&b)/float(len(a|b)) if a or b else 1.0

def rel_from_note_list(notes_low_to_high: List[str]) -> List[int]:
    """Construit l'ordre relatif (pcs) à partir de notes sans octaves, ordonnées grave→aigu."""
    if not notes_low_to_high: return []
    base = NOTE_VALUES[normalize_note(notes_low_to_high[0])]
    rel=[]; prev = base
    for n in notes_low_to_high:
        pc = NOTE_VALUES[normalize_note(n)]
        k=0
        while (pc + 12*k) < prev:
            k += 1
        rel.append((pc + 12*k - base) % 12)
        prev = pc + 12*k
    rel_sorted = sorted(rel)
    return rel_sorted

def real_chord_pcs_and_order(root: str, quality: str) -> Tuple[List[int], List[int], str, int]:
    """Calcul des pcs + ordre relatif pour un symbole d’accord."""
    r = normalize_note(root)
    r_val = NOTE_VALUES[r]
    offs = real_chord_offsets(quality)
    abs_vals = sorted(r_val + o for o in offs)
    pcs = sorted({v % 12 for v in abs_vals})
    base = abs_vals[0]
    rel = [(v - base) % 12 for v in abs_vals]
    return pcs, rel, r, r_val

def rate_symbol_candidate(c:Dict, root:str, quality:str, real_pcs:List[int], real_rel:List[int]) -> Tuple[int,int,int,int,int,int,int,str]:
    """
    Clé de tri (plus petit = mieux) + raison.
    (inv, root_match, bass_match, bal_pref, miss, extra, set_dist, reason)
    """
    inv = 0 if c['rel_order']==real_rel else 1
    root_match = 0 if c['sy_root']==root else 1
    bass_match = 0 if c['root_pc']==NOTE_VALUES[root] else 1
    bal_pref = triad_bal_pref_rank(c)
    cset=set(c['pcs_set']); rset=set(real_pcs)
    miss=len(rset-cset); extra=len(cset-rset)
    set_dist = 0 if (miss==0 and extra==0) else (10 - int(10*jaccard(cset,rset)))
    if miss==0 and extra==0 and inv==0 and root_match==0: reason="exact"
    elif miss==0 and extra==0 and inv==0: reason="exact-ordre"
    elif miss==0 and extra==0: reason="mêmes notes (inversion)"
    else: reason="approx (ensemble diff.)"
    return (inv, root_match, bass_match, bal_pref, miss, extra, set_dist, reason)

def stars_from_symbol_metrics(metrics:Tuple[int,int,int,int,int,int,int,str], pref_bass:bool=True, pref_root:bool=True) -> int:
    inv, root_match, bass_match, bal_pref, miss, extra, set_dist, _ = metrics
    if miss==0 and extra==0 and inv==0 and root_match==0: return 5
    if miss==0 and extra==0 and inv==0: return 4
    if miss==0 and extra==0:
        # si préférences actives, pénaliser un peu la basse/racine
        if (pref_bass and bass_match) or (pref_root and root_match): return 3
        return 4
    if set_dist<=3: return 2
    return 1

def rate_notes_candidate(c:Dict, user_notes:List[str]) -> Tuple[int,int,int,int,int,int,int,str]:
    """Évalue vs liste de notes bass→top. Compare pcs + ordre relatif calculé."""
    user_rel = rel_from_note_list(user_notes)
    user_set = sorted({NOTE_VALUES[n] for n in user_notes})
    inv = 0 if c['rel_order']==user_rel else 1
    cset = c['pcs_set']; miss=len(set(user_set)-set(cset)); extra=len(set(cset)-set(user_set))
    set_dist = 0 if (miss==0 and extra==0) else (10 - int(10*jaccard(set(user_set), set(cset))))
    bal_pref = triad_bal_pref_rank(c)
    bass_pc = NOTE_VALUES[normalize_note(user_notes[0])]
    bass_match = 0 if c['root_pc']==bass_pc else 1
    root_match = 0  # pas imposé
    reason = "exact" if (miss==0 and extra==0 and inv==0) else ("mêmes notes (inversion)" if (miss==0 and extra==0) else "approx (ensemble diff.)")
    return (inv, root_match, bass_match, bal_pref, miss, extra, set_dist, reason)

def stars_from_notes_metrics(metrics:Tuple[int,int,int,int,int,int,int,str]) -> int:
    inv, _, bass_match, _, miss, extra, set_dist, _ = metrics
    if miss==0 and extra==0 and inv==0: return 5
    if miss==0 and extra==0: return 4 if bass_match else 5
    if set_dist<=3: return 2
    return 1

# ---------------------- Exports (JSON / YAML / MD) ----------------------
def mini_yaml(obj, indent=0) -> str:
    sp = '  ' * indent
    if isinstance(obj, dict):
        lines = []
        for k in obj:
            v = obj[k]
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(mini_yaml(v, indent+1))
            else:
                sval = json.dumps(v, ensure_ascii=False)
                lines.append(f"{sp}{k}: {sval}")
        return '\n'.join(lines)
    elif isinstance(obj, list):
        lines = []
        for it in obj:
            if isinstance(it, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(mini_yaml(it, indent+1))
            else:
                sval = json.dumps(it, ensure_ascii=False)
                lines.append(f"{sp}- {sval}")
        return '\n'.join(lines)
    else:
        return f"{sp}{json.dumps(obj, ensure_ascii=False)}"

def export_bundle(data: Dict[str, Any]) -> Dict[str, str]:
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base = f"sy_chord_{ts}"
    paths = {}
    # JSON
    p_json = os.path.join(EXPORT_DIR, base + '.json')
    with open(p_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    paths['json'] = p_json
    # YAML
    p_yaml = os.path.join(EXPORT_DIR, base + '.yaml')
    with open(p_yaml, 'w', encoding='utf-8') as f:
        f.write(mini_yaml(data) + '\n')
    paths['yaml'] = p_yaml
    # MD
    p_md = os.path.join(EXPORT_DIR, base + '.md')
    with open(p_md, 'w', encoding='utf-8') as f:
        f.write(f"# SY CHORD Export — {ts}\n\n")
        q = data.get('query', {})
        f.write("## Requête\n")
        for k,v in q.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Meilleure correspondance\n")
        best = data.get('best')
        if best:
            f.write(f"- Root={best['sy_root']}  Preset={best['sy_type']}  BAL={best['bal']}  Motif={best['bal_pattern']}  |  ⭐ {best.get('stars', '?')}/5\n")
            f.write(f"- Notes: {' '.join(best.get('notes',[]))}\n")
            f.write(f"- Notes+Oct: {' '.join(best.get('notes_oct',[]))}\n\n")
        if data.get('alternatives'):
            f.write("## Alternatives (top)\n")
            for c in data['alternatives'][:20]:
                f.write(f"- ⭐{c.get('stars','?')}/5 | Root={c['sy_root']} Preset={c['sy_type']} BAL={c['bal']} Motif={c['bal_pattern']} | Oct: {' '.join(c.get('notes_oct',[]))}\n")
            f.write('\n')
    paths['md'] = p_md
    return paths

def ensure_readme():
    if not os.path.exists(README):
        with open(README, 'w', encoding='utf-8') as f:
            f.write("# Syntakt SY CHORD — Convertisseur (Pythonista)\n\n")
            f.write("Entrée : nom d’accord (ex. Am) OU liste de notes graves→aigu (ex. E A C).\n")
            f.write("Sortie : Root/Preset/BAL + motif + octaves + étoiles.\n")
    with open(README, 'a', encoding='utf-8') as f:
        f.write(f"\n_Last session_: {datetime.datetime.now().isoformat(timespec='seconds')}\n")

def append_session_md(entry: str):
    with open(RECAPI_MD, 'a', encoding='utf-8') as f:
        f.write(entry + "\n")

# ---------------------- SQLite Logs ----------------------
def init_db():
    conn = sqlite3.connect(LOG_DB)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        mode TEXT, input TEXT,
        anchor_octave INTEGER,
        best_sy_root TEXT, best_sy_type TEXT, best_bal INTEGER,
        best_stars INTEGER, best_notes TEXT, best_notes_oct TEXT
    )
    """)
    conn.commit()
    conn.close()
init_db()

def log_event(mode:str, input_text:str, anchor_octave:int, best: Optional[Dict]):
    conn = sqlite3.connect(LOG_DB)
    cur = conn.cursor()
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    if best:
        cur.execute("""INSERT INTO logs(ts,mode,input,anchor_octave,best_sy_root,best_sy_type,best_bal,best_stars,best_notes,best_notes_oct)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (ts, mode, input_text, int(anchor_octave),
                     best['sy_root'], best['sy_type'], int(best['bal']),
                     int(best.get('stars',0)), ' '.join(best.get('notes',[])), ' '.join(best.get('notes_oct',[]))))
    else:
        cur.execute("""INSERT INTO logs(ts,mode,input,anchor_octave,best_sy_root,best_sy_type,best_bal,best_stars,best_notes,best_notes_oct)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (ts, mode, input_text, int(anchor_octave), None, None, None, None, None, None))
    conn.commit()
    conn.close()

# ---------------------- Matching + Ranking ----------------------
def list_matches_symbol(root:str, quality:str, sy_lib:List[Dict],
                        anchor_octave:int, topk:int, strict:bool,
                        pref_bass:bool, pref_root:bool) -> List[Dict]:
    # cible "réelle"
    rpcs, rrel, rr_norm, rr_val = real_chord_pcs_and_order(root, quality)
    rr_pc = rr_val % 12
    # candidats : même ensemble
    cands = [c for c in sy_lib if set(c['pcs_set'])==set(rpcs)]
    # filtre strict (ordre identique)
    if strict:
        cands = [c for c in cands if c['rel_order']==rrel]
    # tri heuristique (inv, root, bass, bal_pref)
    def _key(c):
        inv = 0 if c['rel_order']==rrel else 1
        root_match = 0 if c['sy_root']==rr_norm else (2 if pref_root else 1)
        bass_match = 0 if c['root_pc']==rr_pc else (2 if pref_bass else 1)
        bal_pref = triad_bal_pref_rank(c)
        return (inv, root_match, bass_match, bal_pref, -c.get('nvoices',4))
    cands.sort(key=_key)
    # enrichir + star rating
    out=[]
    for c in cands[:64]:
        metrics = rate_symbol_candidate(c, rr_norm, quality, rpcs, rrel)
        stars   = stars_from_symbol_metrics(metrics, pref_bass, pref_root)
        v = voices_with_octaves(c, anchor_octave=anchor_octave)
        c2 = dict(c); c2['notes_oct']=[d['label'] for d in v]; c2['metrics']=metrics; c2['stars']=stars
        out.append(c2)
    # order final par étoiles puis métriques
    out.sort(key=lambda c:(-c['stars'], c['metrics']))
    return out[:topk]

def list_matches_notes(user_notes:List[str], sy_lib:List[Dict],
                       anchor_octave:int, topk:int, strict:bool) -> List[Dict]:
    user_set = sorted({NOTE_VALUES[n] for n in user_notes})
    user_rel = rel_from_note_list(user_notes)
    cands = [c for c in sy_lib if set(c['pcs_set'])==set(user_set)]
    if strict:
        cands = [c for c in cands if c['rel_order']==user_rel]
    # tri préférant ordre identique, basse respectée, puis BAL pref
    def _key(c):
        inv = 0 if c['rel_order']==user_rel else 1
        bass_pc = NOTE_VALUES[user_notes[0]]
        bass_match = 0 if c['root_pc']==bass_pc else 1
        bal_pref = triad_bal_pref_rank(c)
        return (inv, bass_match, bal_pref, -c.get('nvoices',4))
    cands.sort(key=_key)
    out=[]
    for c in cands[:64]:
        metrics = rate_notes_candidate(c, user_notes)
        stars   = stars_from_notes_metrics(metrics)
        v = voices_with_octaves(c, anchor_octave=anchor_octave)
        c2 = dict(c); c2['notes_oct']=[d['label'] for d in v]; c2['metrics']=metrics; c2['stars']=stars
        out.append(c2)
    out.sort(key=lambda c:(-c['stars'], c['metrics']))
    return out[:topk]

# ---------------------- TableView custom (title + subtitle) ----------------------
class AltDataSource(object):
    def __init__(self):
        self.items: List[Dict[str,str]] = []

    def tableview_number_of_rows(self, tv, section):
        return len(self.items)

    def tableview_cell_for_row(self, tv, section, row):
        item = self.items[row]
        cell = ui.TableViewCell('subtitle')
        cell.text_label.text = item.get('title','')
        cell.text_label.font = ('<System-Bold>', 14)
        cell.detail_text_label.text = item.get('subtitle','')
        cell.detail_text_label.text_color = GRAY
        cell.detail_text_label.number_of_lines = 2
        return cell

# ---------------------- Overlay Picker ----------------------
class OverlayPicker(ui.View):
    def __init__(self, title: str, items: List[str], on_pick):
        super().__init__(bg_color=(0,0,0,0.25))
        self.items = items
        self.on_pick = on_pick
        self.frame = (0,0,0,0)
        self.flex = 'WH'
        self.container = ui.View(bg_color='white')
        self.container.corner_radius = 10
        self.container.border_color = BORDER
        self.container.border_width = 0.5
        self.add_subview(self.container)

        self.lbl = ui.Label(text=title, alignment=ui.ALIGN_CENTER, font=('<System-Bold>',16))
        self.container.add_subview(self.lbl)

        self.table = ui.TableView()
        self.table.row_height = 40
        self.ds = ui.ListDataSource(items)
        self.ds.action = self._picked
        self.table.data_source = self.ds
        self.table.delegate = self.ds
        self.container.add_subview(self.table)

        self.cancel_btn = ui.Button(title='Annuler')
        self.cancel_btn.action = self._cancel
        self.cancel_btn.tint_color = IOS_BLUE
        self.container.add_subview(self.cancel_btn)

    def layout(self):
        W,H = self.bounds.w, self.bounds.h
        cw, ch = min(360, W-40), min(400, H-160)
        self.container.frame = ((W-cw)/2, (H-ch)/2, cw, ch)
        self.lbl.frame = (10, 10, cw-20, 28)
        self.cancel_btn.frame = (cw-90, 10, 80, 28)
        self.table.frame = (10, 48, cw-20, ch-58)

    def _picked(self, sender):
        idx = sender.selected_row
        if idx is None: return
        val = self.items[idx]
        try:
            self.on_pick(val)
        finally:
            self.close()

    def _cancel(self, sender):
        self.close()

# ---------------------- UI principale ----------------------
class SyntaktApp(object):
    def __init__(self):
        self.sy_lib = build_syntakt_library()
        self.v = self.create_view()
        ensure_readme()
        # état
        self._last_best = None
        self._last_alts = []
        self._last_query = {}
        self._last_opts = {}

    def create_view(self):
        v = ui.View()
        v.name = 'Syntakt - SY CHORD'
        v.background_color = BG

        # Top bar
        self.close_btn = ui.Button(title='✕')
        self.close_btn.action = lambda s: v.close()
        self.close_btn.tint_color = FG

        self.title_lbl = ui.Label(text='Syntakt SY CHORD — Convertisseur', alignment=ui.ALIGN_CENTER, font=('<System-Bold>',18))

        # Entrée libre (accord OU notes)
        self.free_tf = ui.TextField(placeholder='Entrée libre (ex. "Am" ou "E A C")')
        for tf in [self.free_tf]:
            tf.border_width = 1; tf.border_color = BORDER; tf.corner_radius = 6
            tf.autocapitalization_type = ui.AUTOCAPITALIZE_NONE

        # Inputs accord (optionnels si free_tf rempli)
        self.root_tf = ui.TextField(placeholder='Racine (A, C#, Eb...)')
        self.root_tf.border_width = 1; self.root_tf.border_color = BORDER; self.root_tf.corner_radius = 6
        self.root_tf.autocapitalization_type = ui.AUTOCAPITALIZE_ALL

        self.qual_tf = ui.TextField(placeholder='Qualité (m, 7, Maj7, m7b5, add9...)')
        self.qual_tf.border_width = 1; self.qual_tf.border_color = BORDER; self.qual_tf.corner_radius = 6
        self.qual_tf.autocapitalization_type = ui.AUTOCAPITALIZE_NONE

        self.root_pick_btn = ui.Button(title='▼'); self.root_pick_btn.action = self.pick_root
        self.qual_pick_btn = ui.Button(title='▼'); self.qual_pick_btn.action = self.pick_quality

        # Anchor octave
        self.anchor_lbl = ui.Label(text='Octave d’ancrage (C_octave) :', font=('<System>',13))
        self.anchor_tf = ui.TextField(placeholder='3', text='3', alignment=ui.ALIGN_CENTER)
        self.anchor_tf.border_width = 1; self.anchor_tf.border_color = BORDER; self.anchor_tf.corner_radius = 6
        self.anchor_tf.keyboard_type = ui.KEYBOARD_NUMBER_PAD

        # Buttons
        self.convert_btn = ui.Button(title='Analyser'); self.convert_btn.action = self.do_convert
        self.convert_btn.background_color = IOS_BLUE; self.convert_btn.tint_color = 'white'; self.convert_btn.corner_radius = 6

        self.copy_btn = ui.Button(title='Copier'); self.copy_btn.action = self.do_copy
        self.copy_btn.background_color = IOS_GREEN; self.copy_btn.tint_color = 'white'; self.copy_btn.corner_radius = 6

        self.copy_rpb_btn = ui.Button(title='Copy R-P-B'); self.copy_rpb_btn.action = self.do_copy_rpb
        self.copy_rpb_btn.background_color = (0.9,0.6,0.0); self.copy_rpb_btn.tint_color = 'white'; self.copy_rpb_btn.corner_radius = 6

        self.export_btn = ui.Button(title='Exporter'); self.export_btn.action = self.do_export
        self.export_btn.background_color = (0.2, 0.6, 0.9); self.export_btn.tint_color = 'white'; self.export_btn.corner_radius = 6

        # Switches / Options
        self.list_all_sw = ui.Switch(value=False)
        self.strict_sw = ui.Switch(value=False)
        self.pref_bass_sw = ui.Switch(value=True)
        self.pref_root_sw = ui.Switch(value=False)

        self.list_all_lbl = ui.Label(text='Lister tout (Top-K↑)', font=('<System>',13))
        self.strict_lbl = ui.Label(text='Strict (sans inversion)', font=('<System>',13))
        self.pref_bass_lbl = ui.Label(text='Basse = fondamentale (préférence)', font=('<System>',13))
        self.pref_root_lbl = ui.Label(text='Racine identique (préférence)', font=('<System>',13))

        # Help
        self.help_txt = ui.TextView(editable=False, font=('<System>',13), text_color=FG, border_width=0.5, border_color=BORDER, corner_radius=8)
        self.help_txt.text = (
            "Entrée libre: tape un nom d’accord (ex: Am, FMaj7) ou des notes graves→aigu (ex: E A C).\n"
            "Réglages: Strict impose l’ordre exact (sans inversion). Les préférences ajustent le tri.\n"
            "Ancrage: l’octave de référence pour afficher G3–C4–E4. Valeur 3 = C3 ≈ 48 MIDI."
        )

        # Result
        self.result_txt = ui.TextView(editable=False, font=('Menlo',12), text_color=FG, border_width=0.5, border_color=BORDER, corner_radius=8)

        # Alternatives
        self.alt_lbl = ui.Label(text='Alternatives :', font=('<System-Bold>',16))
        self.alt_tv = ui.TableView(row_height=56)
        self.alt_ds = AltDataSource()
        self.alt_tv.data_source = self.alt_ds
        self.alt_tv.delegate = self.alt_ds

        # Add subviews
        for w in (self.close_btn, self.title_lbl,
                  self.free_tf,
                  self.root_tf, self.qual_tf, self.root_pick_btn, self.qual_pick_btn,
                  self.anchor_lbl, self.anchor_tf,
                  self.convert_btn, self.copy_btn, self.copy_rpb_btn, self.export_btn,
                  self.list_all_sw, self.list_all_lbl, self.strict_sw, self.strict_lbl,
                  self.pref_bass_sw, self.pref_bass_lbl, self.pref_root_sw, self.pref_root_lbl,
                  self.help_txt, self.result_txt, self.alt_lbl, self.alt_tv):
            v.add_subview(w)

        # Layout dynamique & safe-area
        def _layout(sender):
            W, H = sender.bounds.w, sender.bounds.h
            top, right, bottom, left = get_safe_insets(sender)
            pad = 16.0
            col = left + pad
            row = top + pad

            # Top bar
            self.close_btn.frame = (col, row, 32, 32)
            self.title_lbl.frame = (col+40, row, W - (col+40) - (pad+right), 32)
            row += 40

            gap = 10
            usable = W - (col + (pad+right))

            # Entrée libre (1 ligne)
            self.free_tf.frame = (col, row, usable, 36)
            row += 36 + gap

            # Inputs Root/Qual + pickers (1 ligne)
            self.root_tf.frame = (col, row, (usable/2) - gap - 30, 36)
            self.root_pick_btn.frame = (self.root_tf.frame[0] + self.root_tf.frame[2] + 6, row, 24, 36)
            self.qual_tf.frame = (col + usable/2 + gap/2, row, (usable/2) - 30, 36)
            self.qual_pick_btn.frame = (self.qual_tf.frame[0] + self.qual_tf.frame[2] + 6, row, 24, 36)
            row += 36 + gap

            # Anchor + Switches (2 colonnes)
            sw_h = 28
            col_mid = col + usable/2
            # Ancrage
            self.anchor_lbl.frame = (col, row, (usable/2) - 60, sw_h)
            self.anchor_tf.frame = (col + (usable/2) - 60, row, 60, sw_h)
            # Préfs (droite)
            self.pref_bass_sw.frame = (col_mid+gap/2, row, 50, sw_h)
            self.pref_bass_lbl.frame = (col_mid+gap/2+52, row, (usable/2)-62, sw_h)
            row2 = row + sw_h + 6
            self.pref_root_sw.frame = (col_mid+gap/2, row2, 50, sw_h)
            self.pref_root_lbl.frame = (col_mid+gap/2+52, row2, (usable/2)-62, sw_h)
            # Ligne switches (gauche)
            self.list_all_sw.frame = (col, row2, 50, sw_h)
            self.list_all_lbl.frame = (col+52, row2, (usable/2)-62, sw_h)
            self.strict_sw.frame = (col, row2 + sw_h + 6, 50, sw_h)
            self.strict_lbl.frame = (col+52, row2 + sw_h + 6, (usable/2)-62, sw_h)
            row = max(row2 + sw_h + sw_h + 6, row + sw_h) + gap

            # Buttons Row
            btn_h = 36
            nbtn = 4
            total_gap = gap*(nbtn-1)
            bw = (usable - total_gap) / nbtn
            self.convert_btn.frame  = (col, row, bw, btn_h)
            self.copy_btn.frame     = (col + (bw+gap), row, bw, btn_h)
            self.copy_rpb_btn.frame = (col + 2*(bw+gap), row, bw, btn_h)
            self.export_btn.frame   = (col + 3*(bw+gap), row, bw, btn_h)
            row += btn_h + gap

            # Help
            help_h = 90 if H >= W else 60
            self.help_txt.frame = (col, row, usable, help_h)
            row += help_h + gap

            # Result
            res_h = 170 if H >= W else 130
            self.result_txt.frame = (col, row, usable, res_h)
            row += res_h + gap

            # Alternatives label + table
            self.alt_lbl.frame = (col, row, 200, 24)
            row += 26
            self.alt_tv.frame = (col, row, usable, H - row - (bottom + pad))

        v.layout = _layout
        return v

    # ---------- Pickers ----------
    def pick_root(self, sender):
        items = ALL_ROOTS[:]
        ov = OverlayPicker('Choisir Racine', items, on_pick=lambda val: setattr(self.root_tf, 'text', val))
        self._present_overlay(ov)

    def pick_quality(self, sender):
        base = sorted(set(list(REAL_TRIADS.keys()) + list(REAL_TETRADS.keys())))
        aliases = ['maj','min','dim','aug','sus2','sus4','7','Maj7','m7','m7b5','add9','madd9','M6','m6']
        items = aliases + base
        ov = OverlayPicker('Choisir Qualité', items, on_pick=lambda val: setattr(self.qual_tf, 'text', val))
        self._present_overlay(ov)

    def _present_overlay(self, ov: ui.View):
        ov.frame = self.v.bounds
        ov.flex = 'WH'
        self.v.add_subview(ov)

    # ---------- Actions ----------
    def do_convert(self, sender):
        # Lecture paramètres
        anchor_txt = (self.anchor_tf.text or '3').strip()
        try:
            anchor = int(anchor_txt)
        except ValueError:
            anchor = 3
        anchor = max(-1, min(7, anchor))  # borne raisonnable

        strict = bool(self.strict_sw.value)
        pref_bass = bool(self.pref_bass_sw.value)
        pref_root = bool(self.pref_root_sw.value)
        list_all = bool(self.list_all_sw.value)
        topk = 30 if list_all else 12

        # Entrée : priorité à free_tf si non vide, sinon root/qual
        free = (self.free_tf.text or '').strip()
        mode = None
        payload = None
        try:
            if free:
                mode, payload = parse_free_input(free)
            else:
                root_in = (self.root_tf.text or '').strip() or 'A'
                qual_in = (self.qual_tf.text or '').strip() or 'm'
                # valider
                _ = normalize_note(root_in); _ = real_chord_offsets(qual_in)
                mode, payload = ('symbol', (normalize_note(root_in), REAL_SYNONYMS.get(qual_in, qual_in)))
        except Exception as e:
            console.hud_alert(f"Erreur entrée: {e}", 'error', 2.0)
            return

        # Matching + ranking
        try:
            if mode=='symbol':
                root, qual = payload
                alts = list_matches_symbol(root, qual, self.sy_lib, anchor, topk, strict, pref_bass, pref_root)
                query_lbl = f"{root} {qual}"
            else:
                user_notes = payload  # graves→aigu (ex: E A C)
                alts = list_matches_notes(user_notes, self.sy_lib, anchor, topk, strict)
                query_lbl = "Notes: " + ' '.join(user_notes)
        except Exception as e:
            self.result_txt.text = 'Erreur : ' + str(e)
            self.alt_ds.items = []; self.alt_tv.reload_data()
            return

        if not alts:
            self.result_txt.text = 'Aucune configuration Syntakt ne reproduit cet accord avec ces contraintes.'
            self.alt_ds.items = []; self.alt_tv.reload_data()
            log_event(mode, free or query_lbl, anchor, None)
            return

        best = alts[0]

        # Résumé principal
        stars = "★"*best['stars'] + "☆"*(5-best['stars'])
        lines = []
        lines.append(f"Entrée : {query_lbl}")
        lines.append("")
        lines.append("Meilleure correspondance Syntakt :")
        lines.append(f"  ⭐  {stars}  ({best['stars']}/5)")
        lines.append(f"  Root    = {best['sy_root']}")
        lines.append(f"  Preset  = {best['sy_type']}")
        lines.append(f"  BAL     = {best['bal']}")
        lines.append(f"  Motif   = {best['bal_pattern']}")
        lines.append(f"  Notes   = {' '.join(best.get('notes',[]))}")
        lines.append(f"  Octaves = {' '.join(best.get('notes_oct',[]))}")
        self.result_txt.text = '\n'.join(lines)

        # Alternatives
        items = []
        for c in alts[1:]:
            s = "★"*c['stars'] + "☆"*(5-c['stars'])
            items.append({
                'title': f"{s}  Root={c['sy_root']}  Preset={c['sy_type']}  BAL={c['bal']}",
                'subtitle': f"Motif={c['bal_pattern']}   Oct={' '.join(c.get('notes_oct',[]))}"
            })
        self.alt_ds.items = items
        self.alt_tv.reload_data()

        # Log + récap
        log_event(mode, free or query_lbl, anchor, best)
        append_session_md(
            f"- {datetime.datetime.now().isoformat(timespec='seconds')} | {query_lbl} → Root={best['sy_root']} Preset={best['sy_type']} BAL={best['bal']} ⭐{best['stars']} | Oct={' '.join(best.get('notes_oct',[]))}"
        )

        # Mémoriser pour Copy / Export
        self._last_best = best
        self._last_alts = alts[1:]
        self._last_query = {'input': free or query_lbl, 'mode': mode}
        self._last_opts = {'strict': strict, 'pref_bass_root': pref_bass, 'pref_root_ident': pref_root, 'anchor_octave': anchor, 'topk': topk}

    def do_copy(self, sender):
        txt = (self.result_txt.text or '').strip()
        if txt:
            clipboard.set(txt)
            console.hud_alert('Copié', 'success', 0.8)

    def do_copy_rpb(self, sender):
        best = getattr(self, '_last_best', None)
        if not best:
            console.hud_alert('Aucun résultat à copier', 'error', 1.0); return
        s = f"Root={best['sy_root']}  Preset={best['sy_type']}  BAL={best['bal']}"
        clipboard.set(s)
        console.hud_alert('Root/Preset/BAL copiés', 'success', 0.8)

    def do_export(self, sender):
        best = getattr(self, '_last_best', None)
        alts = getattr(self, '_last_alts', [])
        query = getattr(self, '_last_query', None)
        opts = getattr(self, '_last_opts', None)
        if not (best and query and opts):
            console.hud_alert("Rien à exporter (analyse d’abord)", 'error', 1.5); return
        data = {'query': query, 'options': opts, 'best': best, 'alternatives': alts}
        paths = export_bundle(data)
        ensure_readme()
        console.hud_alert(f"Exporté : {os.path.basename(paths['json'])}\n{os.path.basename(paths['yaml'])}\n{os.path.basename(paths['md'])}", 'success', 1.5)

    # ---------- Présentation ----------
    def present(self):
        self.v.present('fullscreen')

# ---------------------- Entrée ----------------------
def main():
    app = SyntaktApp()
    app.present()

if __name__ == '__main__':
    main()