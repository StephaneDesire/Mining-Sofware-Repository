"""
04_metrics_rq2.py
-----------------
Calcul des métriques RQ2 : nombre de commentaires et commits après review

RQ2: What types of review comments (e.g., corrective, stylistic, security-related, or testing-related)
are most frequently produced by automated review bots on AI-generated pull requests,
and how are these comments addressed by human developers?

Métriques:
- Catégorisation des commentaires (corrective, style, security, testing)
- Resolution rate (suivi des commits de suivi)
- Sentiment analysis pour évaluer le ton des commentaires
"""

import os
import pandas as pd
import numpy as np
import re


# Paramètres généraux

from paths import FINAL_DIR, RESULTS_DIR



# Mots-clés pour la catégorisation des commentaires
# Ces mots-clés sont basés sur des patterns communs dans les commentaires de review

CORRECTIVE_KEYWORDS = [
    "bug", "error", "fix", "wrong", "incorrect", "mistake", "issue", "problem",
    "broken", "fails", "exception", "crash", "null", "undefined"
]

STYLE_KEYWORDS = [
    "style", "format", "indent", "spacing", "naming", "convention", "lint",
    "pep8", "pylint", "formatting", "whitespace", "semicolon", "brace"
]

SECURITY_KEYWORDS = [
    "security", "vulnerability", "xss", "sql injection", "csrf", "auth",
    "authentication", "authorization", "password", "token", "secret", "key",
    "encrypt", "hash", "sanitize", "escape", "injection"
]

TESTING_KEYWORDS = [
    "test", "testing", "coverage", "unit test", "integration", "mock",
    "assert", "should test", "missing test", "test case", "scenario"
]


def categorize_comment(comment_text):
    """
    Catégorise un commentaire selon son type.
    
    Args:
        comment_text: Texte du commentaire
        
    Returns:
        Liste des catégories détectées
    """
    if pd.isna(comment_text):
        return []
    
    text_lower = str(comment_text).lower()
    categories = []
    
    # Vérification de chaque catégorie
    if any(keyword in text_lower for keyword in CORRECTIVE_KEYWORDS):
        categories.append("corrective")
    
    if any(keyword in text_lower for keyword in STYLE_KEYWORDS):
        categories.append("style")
    
    if any(keyword in text_lower for keyword in SECURITY_KEYWORDS):
        categories.append("security")
    
    if any(keyword in text_lower for keyword in TESTING_KEYWORDS):
        categories.append("testing")
    
    # Si aucune catégorie n'est détectée, on classe comme "other"
    if not categories:
        categories.append("other")
    
    return categories


def analyze_sentiment(comment_text):
    """
    Analyse simple du sentiment d'un commentaire.
    
    Cette fonction utilise des mots-clés simples pour déterminer le ton.
    Une amélioration future serait d'utiliser un modèle NLP plus sophistiqué.
    
    Args:
        comment_text: Texte du commentaire
        
    Returns:
        "positive", "negative", ou "neutral"
    """
    if pd.isna(comment_text):
        return "neutral"
    
    text_lower = str(comment_text).lower()
    
    # Mots-clés positifs
    positive_words = [
        "good", "great", "nice", "excellent", "perfect", "thanks", "thank you",
        "approved", "looks good", "well done", "lgtm"
    ]
    
    # Mots-clés négatifs
    negative_words = [
        "bad", "wrong", "incorrect", "should not", "don't", "cannot", "error",
        "issue", "problem", "concern", "worried", "disappointed"
    ]
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"


# 1. Chargement des données

print("Chargement des données pour RQ2...")

from paths import RAW_DIR

rq2_data = pd.read_parquet(f"{FINAL_DIR}/rq2_data.parquet")
pr_timeline = pd.read_parquet(f"{RAW_DIR}/pr_timeline.parquet")

print(f"Nombre de commentaires sur PRs AI : {len(rq2_data)}")


# 2. Catégorisation des commentaires

print("Catégorisation des commentaires...")

# Application de la catégorisation à chaque commentaire
# Note : on suppose que la colonne "body" contient le texte du commentaire
# Si le nom de la colonne est différent, il faudra l'ajuster
comment_column = "body" if "body" in rq2_data.columns else rq2_data.columns[rq2_data.columns.str.contains("comment|text|content", case=False, na=False)][0] if any(rq2_data.columns.str.contains("comment|text|content", case=False, na=False)) else None

if comment_column is None:
    print("Avertissement : Colonne de commentaire non trouvée, utilisation d'une colonne par défaut")
    comment_column = rq2_data.columns[0]  # Utilisation de la première colonne par défaut

rq2_data["categories"] = rq2_data[comment_column].apply(categorize_comment)
rq2_data["sentiment"] = rq2_data[comment_column].apply(analyze_sentiment)

# Expansion des catégories : un commentaire peut avoir plusieurs catégories
# On crée une ligne par catégorie pour faciliter l'analyse
expanded_rows = []
for idx, row in rq2_data.iterrows():
    if len(row["categories"]) > 0:
        for category in row["categories"]:
            expanded_rows.append({
                "pr_id": row["pr_id"],
                "comment_id": row.get("id", idx),
                "category": category,
                "sentiment": row["sentiment"],
                "has_multiple_categories": len(row["categories"]) > 1
            })
    else:
        expanded_rows.append({
            "pr_id": row["pr_id"],
            "comment_id": row.get("id", idx),
            "category": "other",
            "sentiment": row["sentiment"],
            "has_multiple_categories": False
        })

expanded_df = pd.DataFrame(expanded_rows)


# 3. Calcul des statistiques par catégorie

print("Calcul des statistiques par catégorie...")

category_stats = expanded_df.groupby("category").agg({
    "comment_id": "count",
    "sentiment": lambda x: x.value_counts().to_dict()
}).reset_index()

category_stats.columns = ["category", "count", "sentiment_distribution"]

# Calcul des pourcentages
total_comments = len(rq2_data)
category_stats["percentage"] = (category_stats["count"] / total_comments * 100).round(2)

# Distribution du sentiment par catégorie
sentiment_by_category = expanded_df.groupby(["category", "sentiment"]).size().reset_index(name="count")
sentiment_by_category["percentage"] = sentiment_by_category.groupby("category")["count"].transform(
    lambda x: (x / x.sum() * 100).round(2)
)


# 4. Analyse de la résolution des commentaires

print("Analyse de la résolution des commentaires...")

# On cherche les commits qui suivent les commentaires dans la timeline
# Un commentaire est considéré comme résolu s'il y a un commit après le commentaire
# Cette analyse nécessite de croiser les données avec pr_timeline

# Pour simplifier, on calcule le nombre de commentaires par PR
comments_per_pr = rq2_data.groupby("pr_id").size().reset_index(name="n_comments")

# Jointure avec les données de PR pour voir si elles ont été mergées
pr_clean = pd.read_parquet(f"{FINAL_DIR}/pr_clean.parquet")
comments_per_pr = comments_per_pr.merge(
    pr_clean[["id", "merged"]],
    left_on="pr_id",
    right_on="id",
    how="left"
)

# Taux de merge des PRs avec commentaires
merge_rate_with_comments = comments_per_pr["merged"].mean()


# 5. Sauvegarde des résultats

print("Sauvegarde des résultats...")

# Statistiques par catégorie
category_stats.to_csv(f"{RESULTS_DIR}/rq2_category_stats.csv", index=False)

# Distribution du sentiment par catégorie
sentiment_by_category.to_csv(f"{RESULTS_DIR}/rq2_sentiment_by_category.csv", index=False)

# Résumé des commentaires par PR
comments_per_pr.to_csv(f"{RESULTS_DIR}/rq2_comments_per_pr.csv", index=False)

# Résumé global
summary = {
    "metric": [
        "Total Comments",
        "Unique PRs with Comments",
        "Average Comments per PR",
        "Merge Rate (PRs with comments)"
    ],
    "value": [
        len(rq2_data),
        rq2_data["pr_id"].nunique(),
        len(rq2_data) / rq2_data["pr_id"].nunique() if rq2_data["pr_id"].nunique() > 0 else 0,
        merge_rate_with_comments
    ]
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv(f"{RESULTS_DIR}/rq2_summary.csv", index=False)

print("Métriques RQ2 calculées avec succès.")
print("\nRésumé RQ2:")
print(summary_df.to_string(index=False))
print("\nDistribution par catégorie:")
print(category_stats[["category", "count", "percentage"]].to_string(index=False))
