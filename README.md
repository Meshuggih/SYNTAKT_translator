# SYNTAKT Translator (édition headless)

Ce dépôt fournit une version autonome du traducteur **SY CHORD** pour le Syntakt.
Tous les fichiers du dossier `mtn/data` sont conçus pour être chargés comme modules
*top-level* : aucun import `mtn.data.*` n'est nécessaire.

## Installation et imports

Placez les fichiers du dossier `mtn/data` dans votre environnement cible puis
importez simplement :

```python
from SyntaktTranslatorV3 import Session, format_analysis_fr
from kb_scales import recommend_kb_scale
```

## Utilisation rapide

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
| `topk`             | int   | `12`   | Nombre maximal d'alternatives retournées. |

## Conseiller d'échelles clavier

Pour suggérer des échelles Syntakt adaptées à une progression :

```python
from kb_scales import recommend_kb_scale

progression = ["Gmaj7", "D", "Em7", "Cmaj7"]
print(recommend_kb_scale({0, 2, 4, 7}))  # Jeu direct
```

La fonction `recommend_kb_scale` renvoie une liste triée `(scale_name, root_pc, score)`.

## Scripts utilitaires

- `scripts/smoke_check.py` : vérifie l'intégrité du dossier `mtn/data`, teste les
  imports *top-level* et exécute un appel simple à `Session().analyze("G7")`.
