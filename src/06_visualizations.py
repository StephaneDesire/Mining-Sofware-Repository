"""
06_visualizations.py
-------------------
Visualisations pour les métriques RQ1, RQ2 et RQ3.

Ce script lit les CSVs produits par les scripts metrics (dans `results/tables/`)
et génère des figures dans `results/figures/`.

Usage: python3 src/06_visualizations.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Paramètres généraux

from paths import RESULTS_DIR, FIGURES_DIR, FINAL_DIR


# Configuration du style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# 1. Visualisations RQ1

print("Génération des visualisations RQ1...")

try:
    rq1_summary = pd.read_csv(f"{RESULTS_DIR}/rq1_summary.csv")
    rq1_data = pd.read_parquet(f"{FINAL_DIR}/rq1_data.parquet")
    
    # 1.1. Boxplot de la durée de review : AI vs Human
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    rq1_data_clean = rq1_data[rq1_data["review_duration_hours"].notna() & (rq1_data["review_duration_hours"] >= 0)]
    sns.boxplot(
        data=rq1_data_clean,
        x="author_type",
        y="review_duration_hours",
        ax=axes[0]
    )
    axes[0].set_title("Distribution de la durée de review\n(AI vs Human)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Type d'auteur")
    axes[0].set_ylabel("Durée de review (heures)")
    axes[0].set_yscale('log')  # Échelle logarithmique pour mieux visualiser
    
    # Barplot du taux de merge
    merge_rates = rq1_data.groupby("author_type")["merged"].mean()
    axes[1].bar(merge_rates.index, merge_rates.values, color=['#3498db', '#e74c3c'])
    axes[1].set_title("Taux d'acceptation (merge rate)\n(AI vs Human)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Type d'auteur")
    axes[1].set_ylabel("Taux de merge")
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rq1_review_duration_and_merge_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1.2. Distribution du nombre de commentaires
    fig, ax = plt.subplots(figsize=(10, 6))
    rq1_data_comments = rq1_data[rq1_data["n_comments"].notna()]
    sns.boxplot(
        data=rq1_data_comments,
        x="author_type",
        y="n_comments",
        ax=ax
    )
    ax.set_title("Distribution du nombre de commentaires\n(AI vs Human)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Type d'auteur")
    ax.set_ylabel("Nombre de commentaires")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rq1_comments_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualisations RQ1 générées avec succès.")
    
except FileNotFoundError as e:
    print(f"Fichiers RQ1 non trouvés : {e}")


# 2. Visualisations RQ2

print("Génération des visualisations RQ2...")

try:
    rq2_category_stats = pd.read_csv(f"{RESULTS_DIR}/rq2_category_stats.csv")
    rq2_sentiment = pd.read_csv(f"{RESULTS_DIR}/rq2_sentiment_by_category.csv")
    
    # 2.1. Distribution des catégories de commentaires
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Barplot des catégories
    category_counts = rq2_category_stats.sort_values("count", ascending=False)
    axes[0].barh(category_counts["category"], category_counts["count"], color='steelblue')
    axes[0].set_title("Distribution des catégories de commentaires", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Nombre de commentaires")
    axes[0].set_ylabel("Catégorie")
    
    # Pie chart des pourcentages
    axes[1].pie(
        category_counts["count"],
        labels=category_counts["category"],
        autopct='%1.1f%%',
        startangle=90
    )
    axes[1].set_title("Répartition des catégories (%)", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rq2_category_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.2. Distribution du sentiment par catégorie
    if len(rq2_sentiment) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        sentiment_pivot = rq2_sentiment.pivot(index="category", columns="sentiment", values="count").fillna(0)
        sentiment_pivot.plot(kind='bar', stacked=True, ax=ax, color=['#e74c3c', '#f39c12', '#27ae60'])
        ax.set_title("Distribution du sentiment par catégorie", fontsize=12, fontweight='bold')
        ax.set_xlabel("Catégorie")
        ax.set_ylabel("Nombre de commentaires")
        ax.legend(title="Sentiment")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/rq2_sentiment_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Visualisations RQ2 générées avec succès.")
    
except FileNotFoundError as e:
    print(f"Fichiers RQ2 non trouvés : {e}")


# 3. Visualisations RQ3

print("Génération des visualisations RQ3...")

try:
    rq3_summary = pd.read_csv(f"{RESULTS_DIR}/rq3_summary.csv")
    rq3_data = pd.read_parquet(f"{FINAL_DIR}/rq3_data.parquet")
    
    # 3.1. Boxplot de la durée de review : Closed-loop vs Open-loop
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    rq3_data_clean = rq3_data[rq3_data["review_duration_hours"].notna() & (rq3_data["review_duration_hours"] >= 0)]
    rq3_data_clean["loop_type"] = rq3_data_clean["closed_loop"].map({1: "Closed-loop", 0: "Open-loop"})
    
    sns.boxplot(
        data=rq3_data_clean,
        x="loop_type",
        y="review_duration_hours",
        ax=axes[0]
    )
    axes[0].set_title("Distribution de la durée de review\n(Closed-loop vs Open-loop)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Type de loop")
    axes[0].set_ylabel("Durée de review (heures)")
    axes[0].set_yscale('log')
    
    # Barplot du taux de merge
    merge_rates_loop = rq3_data.groupby("closed_loop")["merged"].mean()
    merge_rates_loop.index = merge_rates_loop.index.map({1: "Closed-loop", 0: "Open-loop"})
    axes[1].bar(merge_rates_loop.index, merge_rates_loop.values, color=['#9b59b6', '#16a085'])
    axes[1].set_title("Taux d'acceptation (merge rate)\n(Closed-loop vs Open-loop)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Type de loop")
    axes[1].set_ylabel("Taux de merge")
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rq3_closed_loop_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.2. Proportion de closed-loop
    fig, ax = plt.subplots(figsize=(8, 6))
    closed_loop_count = (rq3_data["closed_loop"] == 1).sum()
    open_loop_count = (rq3_data["closed_loop"] == 0).sum()
    ax.pie(
        [closed_loop_count, open_loop_count],
        labels=["Closed-loop", "Open-loop"],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#9b59b6', '#16a085']
    )
    ax.set_title("Proportion de Closed-loop vs Open-loop\n(PRs AI uniquement)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rq3_closed_loop_proportion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualisations RQ3 générées avec succès.")
    
except FileNotFoundError as e:
    print(f"Fichiers RQ3 non trouvés : {e}")


# 4. Visualisation comparative globale

print("Génération de la visualisation comparative globale...")

try:
    # Comparaison des trois groupes : AI, Human, Closed-loop, Open-loop
    rq1_data_clean = pd.read_parquet(f"{FINAL_DIR}/rq1_data.parquet")
    rq3_data_clean = pd.read_parquet(f"{FINAL_DIR}/rq3_data.parquet")
    
    # Préparation des données pour la comparaison
    rq1_data_clean["group"] = rq1_data_clean["author_type"].map({"ai": "AI", "human": "Human"})
    rq3_data_clean["group"] = rq3_data_clean["closed_loop"].map({1: "AI Closed-loop", 0: "AI Open-loop"})
    
    # Fusion des données pour la comparaison
    comparison_data = pd.concat([
        rq1_data_clean[["group", "review_duration_hours", "merged"]],
        rq3_data_clean[["group", "review_duration_hours", "merged"]]
    ], ignore_index=True)
    
    comparison_data_clean = comparison_data[
        comparison_data["review_duration_hours"].notna() & 
        (comparison_data["review_duration_hours"] >= 0)
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot comparatif
    order = ["Human", "AI", "AI Open-loop", "AI Closed-loop"]
    sns.boxplot(
        data=comparison_data_clean,
        x="group",
        y="review_duration_hours",
        order=order,
        ax=axes[0]
    )
    axes[0].set_title("Comparaison de la durée de review\n(Tous les groupes)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Groupe")
    axes[0].set_ylabel("Durée de review (heures)")
    axes[0].set_yscale('log')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Barplot comparatif du taux de merge
    merge_rates_comparison = comparison_data.groupby("group")["merged"].mean()
    merge_rates_comparison = merge_rates_comparison.reindex(order, fill_value=0)
    axes[1].bar(range(len(merge_rates_comparison)), merge_rates_comparison.values, color=['#e74c3c', '#3498db', '#16a085', '#9b59b6'])
    axes[1].set_xticks(range(len(merge_rates_comparison)))
    axes[1].set_xticklabels(merge_rates_comparison.index, rotation=45, ha='right')
    axes[1].set_title("Comparaison du taux d'acceptation\n(Tous les groupes)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Taux de merge")
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/comparative_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualisation comparative globale générée avec succès.")
    
except Exception as e:
    print(f"Erreur lors de la génération de la visualisation comparative : {e}")

print("\nToutes les visualisations ont été générées avec succès.")
print(f"Figures sauvegardées dans : {FIGURES_DIR}")
