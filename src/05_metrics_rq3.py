"""
05_metrics_rq3.py
-----------------
Calcul des métriques RQ3 : closed-loop (PR AI uniquement)

RQ3: Does a "closed-loop" bias exist when the AI coding agent and the automated review bot
originate from the same provider, and how does this affect pull request outcomes and review behavior?

Métriques:
- Proportion de PRs avec agents et bots du même provider
- Comparaison des acceptance rates entre closed-loop et open-loop
- Comparaison des review durations entre closed-loop et open-loop
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from cliffs_delta import cliffs_delta


# Paramètres généraux

FINAL_DIR = "../data/final"
RESULTS_DIR = "../results/tables"

os.makedirs(RESULTS_DIR, exist_ok=True)


# 1. Chargement des données

print("Chargement des données pour RQ3...")

rq3_data = pd.read_parquet(f"{FINAL_DIR}/rq3_data.parquet")

print(f"Nombre de PRs AI analysées : {len(rq3_data)}")


# 2. Séparation des groupes closed-loop vs open-loop

print("Séparation des groupes...")

# Groupe closed-loop : PRs AI où l'agent et le bot proviennent du même provider
closed_loop_prs = rq3_data[rq3_data["closed_loop"] == 1].copy()

# Groupe open-loop : PRs AI où l'agent et le bot proviennent de providers différents
open_loop_prs = rq3_data[rq3_data["closed_loop"] == 0].copy()

print(f"PRs Closed-loop : {len(closed_loop_prs)}")
print(f"PRs Open-loop : {len(open_loop_prs)}")


# 3. Calcul des métriques descriptives

print("Calcul des métriques descriptives...")

results = []

# 3.1. Proportion de closed-loop
total_ai_prs = len(rq3_data)
closed_loop_proportion = len(closed_loop_prs) / total_ai_prs if total_ai_prs > 0 else 0

results.append({
    "metric": "closed_loop_proportion",
    "value": closed_loop_proportion,
    "closed_loop_count": len(closed_loop_prs),
    "open_loop_count": len(open_loop_prs),
    "total": total_ai_prs
})

# 3.2. Durée de review (médiane et moyenne)
closed_loop_duration = closed_loop_prs["review_duration_hours"].dropna()
open_loop_duration = open_loop_prs["review_duration_hours"].dropna()

if len(closed_loop_duration) > 0:
    results.append({
        "metric": "review_duration_hours",
        "group": "closed-loop",
        "median": closed_loop_duration.median(),
        "mean": closed_loop_duration.mean(),
        "std": closed_loop_duration.std(),
        "q25": closed_loop_duration.quantile(0.25),
        "q75": closed_loop_duration.quantile(0.75),
        "count": len(closed_loop_duration)
    })

if len(open_loop_duration) > 0:
    results.append({
        "metric": "review_duration_hours",
        "group": "open-loop",
        "median": open_loop_duration.median(),
        "mean": open_loop_duration.mean(),
        "std": open_loop_duration.std(),
        "q25": open_loop_duration.quantile(0.25),
        "q75": open_loop_duration.quantile(0.75),
        "count": len(open_loop_duration)
    })

# 3.3. Taux d'acceptation (merge rate)
closed_loop_merge_rate = closed_loop_prs["merged"].mean() if len(closed_loop_prs) > 0 else 0
open_loop_merge_rate = open_loop_prs["merged"].mean() if len(open_loop_prs) > 0 else 0

results.append({
    "metric": "merge_rate",
    "group": "closed-loop",
    "value": closed_loop_merge_rate,
    "count": len(closed_loop_prs),
    "merged": closed_loop_prs["merged"].sum() if len(closed_loop_prs) > 0 else 0
})

results.append({
    "metric": "merge_rate",
    "group": "open-loop",
    "value": open_loop_merge_rate,
    "count": len(open_loop_prs),
    "merged": open_loop_prs["merged"].sum() if len(open_loop_prs) > 0 else 0
})

# 3.4. Nombre de commentaires
closed_loop_comments = closed_loop_prs["n_comments"].dropna()
open_loop_comments = open_loop_prs["n_comments"].dropna()

if len(closed_loop_comments) > 0:
    results.append({
        "metric": "n_comments",
        "group": "closed-loop",
        "median": closed_loop_comments.median(),
        "mean": closed_loop_comments.mean(),
        "std": closed_loop_comments.std(),
        "count": len(closed_loop_comments)
    })

if len(open_loop_comments) > 0:
    results.append({
        "metric": "n_comments",
        "group": "open-loop",
        "median": open_loop_comments.median(),
        "mean": open_loop_comments.mean(),
        "std": open_loop_comments.std(),
        "count": len(open_loop_comments)
    })


# 4. Tests statistiques

print("Calcul des tests statistiques...")

# 4.1. Test de Mann-Whitney U pour la durée de review
if len(closed_loop_duration) > 0 and len(open_loop_duration) > 0:
    u_stat, p_value_duration = stats.mannwhitneyu(
        closed_loop_duration,
        open_loop_duration,
        alternative='two-sided'
    )
    
    # Calcul de l'effect size avec Cliff's delta
    d, d_interpretation = cliffs_delta(closed_loop_duration, open_loop_duration)
    
    results.append({
        "metric": "review_duration_hours",
        "test": "Mann-Whitney U",
        "statistic": u_stat,
        "p_value": p_value_duration,
        "effect_size": d,
        "effect_interpretation": d_interpretation
    })

# 4.2. Test du chi² pour le taux de merge
# Table de contingence : merged vs not merged pour closed-loop vs open-loop
contingency_table = pd.crosstab(
    rq3_data["closed_loop"],
    rq3_data["merged"]
)

if contingency_table.size > 0:
    chi2, p_value_merge, dof, expected = stats.chi2_contingency(contingency_table)
    
    results.append({
        "metric": "merge_rate",
        "test": "Chi-square",
        "statistic": chi2,
        "p_value": p_value_merge,
        "degrees_of_freedom": dof
    })

# 4.3. Test de Mann-Whitney U pour le nombre de commentaires
if len(closed_loop_comments) > 0 and len(open_loop_comments) > 0:
    u_stat_comments, p_value_comments = stats.mannwhitneyu(
        closed_loop_comments,
        open_loop_comments,
        alternative='two-sided'
    )
    
    d_comments, d_comments_interpretation = cliffs_delta(closed_loop_comments, open_loop_comments)
    
    results.append({
        "metric": "n_comments",
        "test": "Mann-Whitney U",
        "statistic": u_stat_comments,
        "p_value": p_value_comments,
        "effect_size": d_comments,
        "effect_interpretation": d_comments_interpretation
    })


# 5. Sauvegarde des résultats

print("Sauvegarde des résultats...")

# Conversion en DataFrame pour faciliter l'export
results_df = pd.DataFrame(results)

# Sauvegarde en CSV
results_df.to_csv(f"{RESULTS_DIR}/rq3_metrics.csv", index=False)

# Sauvegarde d'un résumé formaté
summary = {
    "Metric": [],
    "Closed_Loop_Median": [],
    "Open_Loop_Median": [],
    "Closed_Loop_Mean": [],
    "Open_Loop_Mean": [],
    "P_Value": [],
    "Effect_Size": []
}

# Proportion de closed-loop
summary["Metric"].append("Closed-Loop Proportion")
summary["Closed_Loop_Median"].append(f"{closed_loop_proportion:.2%}")
summary["Open_Loop_Median"].append("N/A")
summary["Closed_Loop_Mean"].append("N/A")
summary["Open_Loop_Mean"].append("N/A")
summary["P_Value"].append("N/A")
summary["Effect_Size"].append("N/A")

# Durée de review
if len(closed_loop_duration) > 0 and len(open_loop_duration) > 0:
    summary["Metric"].append("Review Duration (hours)")
    summary["Closed_Loop_Median"].append(f"{closed_loop_duration.median():.2f}")
    summary["Open_Loop_Median"].append(f"{open_loop_duration.median():.2f}")
    summary["Closed_Loop_Mean"].append(f"{closed_loop_duration.mean():.2f}")
    summary["Open_Loop_Mean"].append(f"{open_loop_duration.mean():.2f}")
    summary["P_Value"].append(f"{p_value_duration:.4f}")
    summary["Effect_Size"].append(f"{d:.3f} ({d_interpretation})")
else:
    summary["Metric"].append("Review Duration (hours)")
    summary["Closed_Loop_Median"].append("N/A")
    summary["Open_Loop_Median"].append("N/A")
    summary["Closed_Loop_Mean"].append("N/A")
    summary["Open_Loop_Mean"].append("N/A")
    summary["P_Value"].append("N/A")
    summary["Effect_Size"].append("N/A")

# Taux de merge
summary["Metric"].append("Merge Rate")
summary["Closed_Loop_Median"].append(f"{closed_loop_merge_rate:.2%}")
summary["Open_Loop_Median"].append(f"{open_loop_merge_rate:.2%}")
summary["Closed_Loop_Mean"].append("N/A")
summary["Open_Loop_Mean"].append("N/A")
summary["P_Value"].append(f"{p_value_merge:.4f}" if 'p_value_merge' in locals() else "N/A")
summary["Effect_Size"].append("N/A")

# Nombre de commentaires
if len(closed_loop_comments) > 0 and len(open_loop_comments) > 0:
    summary["Metric"].append("Number of Comments")
    summary["Closed_Loop_Median"].append(f"{closed_loop_comments.median():.2f}")
    summary["Open_Loop_Median"].append(f"{open_loop_comments.median():.2f}")
    summary["Closed_Loop_Mean"].append(f"{closed_loop_comments.mean():.2f}")
    summary["Open_Loop_Mean"].append(f"{open_loop_comments.mean():.2f}")
    summary["P_Value"].append(f"{p_value_comments:.4f}")
    summary["Effect_Size"].append(f"{d_comments:.3f} ({d_comments_interpretation})")
else:
    summary["Metric"].append("Number of Comments")
    summary["Closed_Loop_Median"].append("N/A")
    summary["Open_Loop_Median"].append("N/A")
    summary["Closed_Loop_Mean"].append("N/A")
    summary["Open_Loop_Mean"].append("N/A")
    summary["P_Value"].append("N/A")
    summary["Effect_Size"].append("N/A")

summary_df = pd.DataFrame(summary)
summary_df.to_csv(f"{RESULTS_DIR}/rq3_summary.csv", index=False)

print("Métriques RQ3 calculées avec succès.")
print("\nRésumé RQ3:")
print(summary_df.to_string(index=False))
