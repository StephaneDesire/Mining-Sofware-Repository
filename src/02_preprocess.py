"""
02_preprocess.py
----------------
Préprocessing des données pour les analyses RQ1, RQ2 et RQ3.

Ce script effectue les transformations nécessaires sur les données filtrées :
- Nettoyage des valeurs manquantes
- Création de variables catégorielles
- Préparation des données pour les analyses statistiques
"""

import os
import pandas as pd
import numpy as np


# Paramètres généraux

INTER_DIR = "../data/intermediate"
FINAL_DIR = "../data/final"

os.makedirs(FINAL_DIR, exist_ok=True)


# 1. Chargement des données intermédiaires

print("Chargement des données intermédiaires...")

pr = pd.read_parquet(f"{INTER_DIR}/pr_filtered_enriched.parquet")
pr_comments = pd.read_parquet("../data/raw/pr_comments.parquet")
pr_reviews = pd.read_parquet("../data/raw/pr_reviews.parquet")
pr_timeline = pd.read_parquet("../data/raw/pr_timeline.parquet")

print(f"Nombre de PRs chargées : {len(pr)}")


# 2. Nettoyage des données

print("Nettoyage des données...")

# 2.1. Gestion des valeurs manquantes pour review_duration_hours
# On filtre les PRs qui n'ont pas été fermées (closed_at manquant)
# ou qui ont une durée négative (erreur de données)
pr_clean = pr[
    (pr["review_duration_hours"].notna()) & 
    (pr["review_duration_hours"] >= 0)
].copy()

print(f"PRs après nettoyage des durées : {len(pr_clean)}")

# 2.2. Création de variables catégorielles pour les analyses
# Type de reviewer : déterminé par la présence de reviews automatisées
# On considère qu'une PR est reviewée par un bot si elle a des reviews automatisées
pr_reviews_auto = pr_reviews[pr_reviews["state"] == "APPROVED"].copy()
# Note : on pourrait améliorer cette logique en analysant les reviewers

# 2.3. Catégorisation des PRs selon leur statut
# PRs mergées vs non-mergées
pr_clean["status"] = pr_clean["merged"].map({1: "merged", 0: "closed"})

# 2.4. Création de groupes pour les comparaisons statistiques
# Groupe 1 : PRs AI reviewées par bots (à déterminer via reviews)
# Groupe 2 : PRs AI reviewées par humains
# Groupe 3 : PRs humaines (baseline)

# Détermination du type de reviewer pour chaque PR (bot / human / none)
# On analyse à la fois les reviews et les commentaires : si au moins un reviewer
# (ou un auteur de review/commentaire) correspond à un mot-clé de bot, on marque
# la PR comme reviewée par un bot. Sinon si des reviews/commentaires existent,
# on marque 'human'. Sinon 'none'.
BOT_KEYWORDS = [
    "copilot", "cursor", "codex", "claude", "devin"
]

def _pick_login_col(df, candidates):
    """Retourne le nom de colonne à utiliser pour le login parmi une liste de candidats."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


print("Détection du type de reviewer (bot / human / none)...")

# Choix des colonnes probables
pr_reviews_login_col = _pick_login_col(pr_reviews, ["author_login", "user_login", "login", "reviewer_login", "actor_login"])
pr_comments_login_col = _pick_login_col(pr_comments, ["author_login", "user_login", "login", "commenter_login", "actor_login", "user"])

# Colonne de liaison vers la PR
pr_id_col_reviews = "pr_id" if "pr_id" in pr_reviews.columns else ("pull_request_id" if "pull_request_id" in pr_reviews.columns else None)
pr_id_col_comments = "pr_id" if "pr_id" in pr_comments.columns else ("pull_request_id" if "pull_request_id" in pr_comments.columns else None)

# Map pour stocker le type de reviewer par PR id
reviewer_type_map = {}

# 1) Analyse des reviews
if pr_id_col_reviews is not None and pr_reviews_login_col is not None:
    # Normaliser les logins en minuscules
    pr_reviews[pr_reviews_login_col] = pr_reviews[pr_reviews_login_col].fillna("").astype(str)

    # Regrouper les logins par PR
    grouped = pr_reviews.groupby(pr_id_col_reviews)[pr_reviews_login_col].apply(list)
    for pr_id, logins in grouped.items():
        # vérifier présence d'un mot-clé de bot
        if any(any(k in (l or "").lower() for k in BOT_KEYWORDS) for l in logins):
            reviewer_type_map[pr_id] = "bot"
        else:
            reviewer_type_map.setdefault(pr_id, "human")

# 2) Analyse des commentaires (complémentaire)
if pr_id_col_comments is not None and pr_comments_login_col is not None:
    pr_comments[pr_comments_login_col] = pr_comments[pr_comments_login_col].fillna("").astype(str)
    grouped_c = pr_comments.groupby(pr_id_col_comments)[pr_comments_login_col].apply(list)
    for pr_id, logins in grouped_c.items():
        if any(any(k in (l or "").lower() for k in BOT_KEYWORDS) for l in logins):
            reviewer_type_map[pr_id] = "bot"
        else:
            reviewer_type_map.setdefault(pr_id, "human")

# 3) Application du mapping au dataframe pr_clean
def map_reviewer_type(pr_id):
    return reviewer_type_map.get(pr_id, "none")

pr_clean["reviewer_type"] = pr_clean["id"].apply(map_reviewer_type)

print("Types de reviewer détectés :", pr_clean["reviewer_type"].value_counts().to_dict())


# 3. Préparation des données pour RQ1

print("Préparation des données pour RQ1...")

# RQ1 compare les durées de review et taux d'acceptation
# entre PRs AI reviewées par bots vs PRs reviewées par humains
rq1_data = pr_clean[
    ["id", "author_type", "review_duration_hours", "merged", "status", "n_comments", "reviewer_type"]
].copy()

# Ajout d'un flag pour les PRs AI
rq1_data["is_ai"] = (rq1_data["author_type"] == "ai").astype(int)


# 4. Préparation des données pour RQ2

print("Préparation des données pour RQ2...")

# RQ2 analyse les commentaires sur les PRs AI
# Jointure avec les commentaires pour analyser leur contenu
rq2_pr_comments = pr_comments.merge(
    pr_clean[["id", "author_type"]],
    left_on="pr_id",
    right_on="id",
    how="inner"
)

# Filtrage pour ne garder que les commentaires sur les PRs AI
rq2_data = rq2_pr_comments[
    rq2_pr_comments["author_type"] == "ai"
].copy()

print(f"Nombre de commentaires sur PRs AI : {len(rq2_data)}")


# 5. Préparation des données pour RQ3

print("Préparation des données pour RQ3...")

# RQ3 analyse uniquement les PRs AI avec closed-loop vs open-loop
rq3_data = pr_clean[
    pr_clean["author_type"] == "ai"
].copy()

# Le flag closed_loop a déjà été calculé dans 01_load_filter.py
# On crée une variable catégorielle pour faciliter les analyses
rq3_data["loop_type"] = rq3_data["closed_loop"].map({1: "closed-loop", 0: "open-loop"})

print(f"PRs AI - Closed-loop : {rq3_data['closed_loop'].sum()}")
print(f"PRs AI - Open-loop : {(rq3_data['closed_loop'] == 0).sum()}")


# 6. Sauvegarde des datasets finaux

print("Sauvegarde des datasets finaux...")

# Dataset principal nettoyé
pr_clean.to_parquet(f"{FINAL_DIR}/pr_clean.parquet")

# Datasets pour chaque RQ
rq1_data.to_parquet(f"{FINAL_DIR}/rq1_data.parquet")
rq2_data.to_parquet(f"{FINAL_DIR}/rq2_data.parquet")
rq3_data.to_parquet(f"{FINAL_DIR}/rq3_data.parquet")

print("Préprocessing terminé avec succès.")

