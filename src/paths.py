"""
paths.py
--------
Centralise les chemins de fichiers utilisés par les scripts.

Importer depuis les scripts avec :
    from paths import RAW_DIR, INTER_DIR, FINAL_DIR, RESULTS_DIR, FIGURES_DIR

Le module crée également les dossiers s'ils n'existent pas.
"""

import os

# Dossiers de données
RAW_DIR = "data/raw"
INTER_DIR = "data/intermediate"
FINAL_DIR = "data/final"

# Dossiers de résultats
RESULTS_DIR = "results/tables"
FIGURES_DIR = "results/figures"

# S'assurer que les dossiers existent
for d in [RAW_DIR, INTER_DIR, FINAL_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)
