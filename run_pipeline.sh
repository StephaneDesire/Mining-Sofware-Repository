#!/bin/bash

# run_pipeline.sh
# ---------------
# Script pour exécuter l'ensemble du pipeline d'analyse AIDev (MSR Challenge).
#
# Ce script exécute les étapes suivantes :
# 1. Chargement et filtrage des données
# 2. Préprocessing des données
# 3. Calcul des métriques RQ1, RQ2, RQ3
# 4. Génération des visualisations
#
# Usage: bash run_pipeline.sh

set -e  # Arrêter le script en cas d'erreur

echo "=========================================="
echo "Pipeline d'analyse AIDev (MSR Challenge)"
echo "=========================================="
echo ""

# Vérification que Python est disponible
if ! command -v python3 &> /dev/null; then
    echo "Erreur : Python3 n'est pas installé ou n'est pas dans le PATH"
    exit 1
fi

# Vérification que les dépendances sont installées
echo "Vérification des dépendances..."
python3 -c "import pandas, numpy, scipy, matplotlib, seaborn, cliffs_delta" 2>/dev/null || {
    echo "Erreur : Certaines dépendances Python sont manquantes"
    echo "Veuillez installer les dépendances avec : pip install -r requirements.txt"
    exit 1
}
echo "Dépendances OK"
echo ""

# Changement vers le répertoire src pour l'exécution
cd "$(dirname "$0")/src" || exit 1

# Étape 1 : Chargement et filtrage des données
echo "=========================================="
echo "Étape 1/6 : Chargement et filtrage"
echo "=========================================="
python3 01_load_filter.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'étape 1"
    exit 1
fi
echo ""

# Étape 2 : Préprocessing
echo "=========================================="
echo "Étape 2/6 : Préprocessing"
echo "=========================================="
python3 02_preprocess.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'étape 2"
    exit 1
fi
echo ""

# Étape 3 : Métriques RQ1
echo "=========================================="
echo "Étape 3/6 : Calcul des métriques RQ1"
echo "=========================================="
python3 03_metrics_rq1.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'étape 3"
    exit 1
fi
echo ""

# Étape 4 : Métriques RQ2
echo "=========================================="
echo "Étape 4/6 : Calcul des métriques RQ2"
echo "=========================================="
python3 04_metrics_rq2.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'étape 4"
    exit 1
fi
echo ""

# Étape 5 : Métriques RQ3
echo "=========================================="
echo "Étape 5/6 : Calcul des métriques RQ3"
echo "=========================================="
python3 05_metrics_rq3.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'étape 5"
    exit 1
fi
echo ""

# Étape 6 : Visualisations
echo "=========================================="
echo "Étape 6/6 : Génération des visualisations"
echo "=========================================="
python3 06_visualizations.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'étape 6"
    exit 1
fi
echo ""

# Retour au répertoire racine
cd ..

echo "=========================================="
echo "Pipeline terminé avec succès !"
echo "=========================================="
echo ""
echo "Résultats disponibles dans :"
echo "  - Tables : results/tables/"
echo "  - Figures : results/figures/"
echo ""

