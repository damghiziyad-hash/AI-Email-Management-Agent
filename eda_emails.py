"""
Phase 4 — EDA (Exploratory Data Analysis)

Usage:
    python eda_emails.py emails_clean.csv

Attend les colonnes: email_id, sender, subject_clean, body_clean, text_length, label
"""

import sys
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

LABELS = {0: "NORMAL", 1: "IMPORTANT", 2: "SPAM"}

# Mots trop frequents en francais pour etre informatifs (stopwords basiques)
STOPWORDS = set("""
le la les de des du un une et en a au aux ce cet cette ces pour dans que qui
est sont sur avec par se ne pas vous nous votre vos notre nos il elle ils
elles je tu on son sa ses leur leurs plus moins tres bien tout tous toute
url email d l n c j s qu
""".split())


def word_frequencies(texts, top_n=10):
    words = " ".join(texts).split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return Counter(words).most_common(top_n)


def domain_frequencies(senders, top_n=10):
    domains = [s.split("@")[-1] for s in senders if "@" in s]
    return Counter(domains).most_common(top_n)


def link_presence_rate(texts):
    """% d'emails contenant le token URL (injecte en Phase 3)."""
    has_link = [1 if "url" in t.split() else 0 for t in texts]
    return sum(has_link) / len(has_link) * 100 if has_link else 0


def run_eda(df: pd.DataFrame):
    df["label_name"] = df["label"].map(LABELS)
    df["full_text"] = df["subject_clean"].fillna("") + " " + df["body_clean"].fillna("")

    print("=" * 50)
    print("1. DISTRIBUTION DES CLASSES")
    print("=" * 50)
    dist = df["label_name"].value_counts()
    dist_pct = df["label_name"].value_counts(normalize=True) * 100
    for label in dist.index:
        print(f"{label:10s}: {dist[label]:3d} emails  ({dist_pct[label]:.1f}%)")

    print()
    print("=" * 50)
    print("2. LONGUEUR MOYENNE DU TEXTE PAR CLASSE")
    print("=" * 50)
    print(df.groupby("label_name")["text_length"].agg(["mean", "min", "max"]).round(1))

    print()
    print("=" * 50)
    print("3. MOTS LES PLUS FREQUENTS PAR CLASSE")
    print("=" * 50)
    for label_val, label_name in LABELS.items():
        subset = df[df["label"] == label_val]["full_text"]
        top_words = word_frequencies(subset, top_n=8)
        print(f"\n{label_name}:")
        for word, count in top_words:
            print(f"   {word:20s} {count}")

    print()
    print("=" * 50)
    print("4. DOMAINES D'EXPEDITEURS LES PLUS FREQUENTS")
    print("=" * 50)
    top_domains = domain_frequencies(df["sender"], top_n=8)
    for domain, count in top_domains:
        print(f"   {domain:30s} {count}")

    print()
    print("=" * 50)
    print("5. PRESENCE DE LIENS PAR CLASSE")
    print("=" * 50)
    for label_val, label_name in LABELS.items():
        subset = df[df["label"] == label_val]["full_text"]
        rate = link_presence_rate(subset)
        print(f"{label_name:10s}: {rate:.1f}% des emails contiennent un lien")

    return df


def make_plots(df: pd.DataFrame, output_path="eda_plots.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Graphique 1: distribution des classes
    dist = df["label_name"].value_counts()
    axes[0].bar(dist.index, dist.values, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("Distribution des classes")
    axes[0].set_ylabel("Nombre d'emails")

    # Graphique 2: longueur moyenne par classe
    avg_len = df.groupby("label_name")["text_length"].mean()
    axes[1].bar(avg_len.index, avg_len.values, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_title("Longueur moyenne du texte")
    axes[1].set_ylabel("Caracteres")

    # Graphique 3: % emails avec lien par classe
    link_rates = []
    for label_val in LABELS:
        subset = df[df["label"] == label_val]["full_text"]
        link_rates.append(link_presence_rate(subset))
    axes[2].bar(list(LABELS.values()), link_rates, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[2].set_title("% d'emails contenant un lien")
    axes[2].set_ylabel("%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    print(f"\nGraphiques sauvegardes: {output_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python eda_emails.py emails_clean.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    df = run_eda(df)
    make_plots(df)


if __name__ == "__main__":
    main()
