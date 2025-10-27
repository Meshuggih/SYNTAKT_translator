# SYNTAKT Translator (édition headless)

Ce dépôt fournit une version autonome du traducteur **SY CHORD** pour le Syntakt.
Tous les fichiers du dossier `mtn/data` sont conçus pour être chargés comme modules
*top-level* ; utilisez uniquement les imports plats présentés ci-dessous.

## Installation et imports

Copiez les six fichiers plats (`SyntaktTranslatorV3.py`, `syntakt_core.py`,
`kb_scales.py`, `Syntakt.csv`, `syntakt_documentation.json`, `__init__.py`) dans
votre environnement cible. Ensuite, utilisez exclusivement les imports
*top-level* suivants :

```python
from SyntaktTranslatorV3 import analyze, voicing_for, Session, format_analysis_fr
from kb_scales import recommend_kb_scale
```

## Utilisation rapide

### API haut-niveau

```python
from SyntaktTranslatorV3 import analyze

print(analyze("Am")["copy_line"])
# Root=A  Preset=minor  BAL=96

best = analyze("E A C")["best"]
print(best)
# {'root': 'A', 'preset': 'minor', 'bal': 106, 'voicing': 'C4 E4 A4', ...}
```

La fonction `analyze()` encapsule `normalize_input()` et `rank_presets()` puis
retourne une structure JSON-friendly contenant :

- `best` : le preset prioritaire avec `root`, `preset`, `bal`, `voicing` (grave →
  aigu) et les métadonnées `labels`, `metrics`, `stars` (0–5).
- `alternatives` : les autres candidats dans l'ordre de `rank_presets()` (top-k).
- `candidates` : même contenu que `alternatives` mais en incluant le meilleur
  preset (pour compatibilité).
- `copy_line` : la ligne copiable `Root=…  Preset=…  BAL=…` identique à celle du
  meilleur preset.
- `stars` : la note 0–5 associée au meilleur preset.

L'ancrage d'octave par défaut place `C3` (≈ MIDI 48) comme référence pour le
voicing. Les notes sont ordonnées strictement du grave vers l'aigu et
normalisées dans une plage compacte. En l'absence de préférence explicite
(`prefer_flats=None`), l'orthographe privilégie les bémols lorsque la racine du
preset comporte un `b` (et aucun `#`). Pour les triades en 2ᵉ renversement,
`analyze()` applique un tie-break supplémentaire en faveur des presets BAL 106
(`•*••`) lorsqu'ils partagent le même score et le même ensemble de hauteurs que
les autres candidats. Les symboles enrichis tels que `Em6` sont désormais
reconnus nativement par le parseur.

```python
from SyntaktTranslatorV3 import voicing_for, analyze

payload = analyze("Em6")
best_preset = payload["best"]
print(voicing_for(payload["candidates"][0], anchor_octave=4))
```

L'appel ci-dessus illustre comment réutiliser un preset (par exemple via un
cache externe) pour recalculer un voicing avec une autre octave d'ancrage ou une
préférence enharmonique spécifique.

Analyse d'un symbole :

```python
session = Session()
result = session.analyze("G7")
print(format_analysis_fr(result))
```

Sortie typique :

```
Résultat :
- Réglage recommandé : Root=G Preset=7 BAL=96 (••••)
- Notes jouées : G3, B3, D4, F4
- Qualité du match : exact (5★)
- Alternatives top-k :
  • 2★ Root=G Preset=major BAL=96 (••••)
  • 1★ Root=G Preset=M6 BAL=96 (••••)
  • 1★ Root=G Preset=Maj7 BAL=96 (••••)
```

Analyse d'une liste de notes (grave → aigu) :

```python
session = Session()
result = session.analyze("E A C", anchor_octave=2)
for alt in result["alternatives"]:
    print(alt["stars"], alt["copy_line"])
```

Dans cette configuration, la ligne copiable du meilleur candidat reste accessible
via `result["best"]["copy_line"]` et l'ordre grave→aigu est préservé dans
`result["best"]["notes_oct"]`.

Chaque analyse retourne un dictionnaire contenant :

- `best` : la recommandation principale avec `copy_line` (`Root=… Preset=… BAL=…`).
- `alternatives` : la liste triée des presets top-k (étoiles décroissantes).
- `copy_lines` : raccourcis textuels copiables.
- `chord_pcs` : ensemble des classes de notes détectées.
- `error` : message explicite en cas d'entrée invalide.

## Options prises en charge

| Option             | Type  | Défaut | Description |
|--------------------|-------|--------|-------------|
| `strict`           | bool  | `False`| Ignore les correspondances à ≤2★. |
| `pref_bass_root`   | bool  | `True` | Favorise les presets dont la basse est la fondamentale. |
| `pref_root_ident`  | bool  | `False`| Favorise les presets partageant le même nom de racine. |
| `anchor_octave`    | int   | `3`    | Octave de référence (C3 ≈ 48 MIDI) pour le voicing renvoyé. |
| `prefer_flats`     | bool  | `None` | Force l'orthographe (bémols ou dièses) des voicings retournés. |
| `topk`             | int   | `12`   | Nombre maximal d'alternatives retournées. |

## Conseiller d'échelles clavier

Pour suggérer des échelles Syntakt adaptées à une progression :

```python
from kb_scales import recommend_kb_scale

progression = ["Gmaj7", "D", "Em7", "Cmaj7"]
print(recommend_kb_scale({0, 2, 4, 7}))  # Jeu direct
```

La fonction `recommend_kb_scale` renvoie une liste triée `(scale_name, root_pc, score)`.

## CLI

L'exécution directe du module fournit un point d'entrée simple pour tester une
analyse sans écrire de script :

```bash
python -m SyntaktTranslatorV3 "Am"
python -m SyntaktTranslatorV3 "E A C"
python -m SyntaktTranslatorV3 "Em6"
```

Chaque appel affiche le JSON complet (compatible avec l'API `analyze()`) puis un
résumé francophone compact basé sur `format_analysis_fr`.

## Scripts utilitaires

- `scripts/smoke_check.py` : vérifie l'intégrité du dossier `mtn/data`, teste les
  imports *top-level* et exécute un appel simple à `Session().analyze("G7")`.
