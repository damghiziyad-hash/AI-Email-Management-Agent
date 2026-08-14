"""
Phase 5 — NLP : transformer le texte en features numeriques.

Usage:
    python nlp_features.py emails_clean.csv

Produit:
    - tfidf_vectorizer.joblib (le vectorizer entraine, reutilisable)
    - features_preview.csv (apercu des features numeriques additionnelles)

NOTE IMPORTANTE:
    Ici le TF-IDF est fit sur TOUT le dataset, uniquement a but exploratoire
    (voir le vocabulaire, les mots discriminants). En Phase 6, le vectorizer
    sera fit UNIQUEMENT sur le train set (dans un Pipeline sklearn) pour
    eviter la fuite de donnees (data leakage) vers le test set.
"""

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

LABELS = {0: "NORMAL", 1: "IMPORTANT", 2: "SPAM"}

# Stopwords francais basiques a exclure du TF-IDF (mots trop frequents, non informatifs)
FRENCH_STOPWORDS = [
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "a", "au", "aux",
    "ce", "cet", "cette", "ces", "pour", "dans", "que", "qui", "est", "sont", "sur",
    "avec", "par", "se", "ne", "pas", "vous", "nous", "votre", "vos", "notre", "nos",
    "il", "elle", "ils", "elles", "je", "tu", "on", "son", "sa", "ses", "leur",
    "leurs", "plus", "moins", "tres", "bien", "tout", "tous", "toute", "d", "l",
    "n", "c", "j", "s", "qu", "etre", "avoir",
]


def build_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features numeriques simples, en plus du TF-IDF."""
    full_text = df["subject_clean"].fillna("") + " " + df["body_clean"].fillna("")

    features = pd.DataFrame()
    features["text_length"] = df["text_length"]
    features["num_words"] = full_text.str.split().apply(len)
    features["num_links"] = full_text.str.split().apply(lambda w: w.count("url"))
    features["num_email_mentions"] = full_text.str.split().apply(lambda w: w.count("email"))
    features["sender_domain_len"] = df["sender"].str.split("@").str[-1].str.len()

    return features


def run_tfidf(df: pd.DataFrame, max_features=300, ngram_range=(1, 2)):
    full_text = (df["subject_clean"].fillna("") + " " + df["body_clean"].fillna("")).tolist()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,  # unigrams + bigrams (ex: "cliquez ici")
        min_df=1,
        stop_words=FRENCH_STOPWORDS,
    )
    tfidf_matrix = vectorizer.fit_transform(full_text)

    print("=" * 50)
    print("RESULTATS TF-IDF")
    print("=" * 50)
    print(f"Taille du vocabulaire : {len(vectorizer.vocabulary_)}")
    print(f"Shape de la matrice   : {tfidf_matrix.shape}  (emails x features)")

    # Mots les plus discriminants PAR CLASSE (moyenne du score tf-idf)
    feature_names = np.array(vectorizer.get_feature_names_out())
    print("\nTop termes TF-IDF par classe (moyenne des scores):")
    for label_val, label_name in LABELS.items():
        mask = (df["label"] == label_val).values
        mean_scores = tfidf_matrix[mask].mean(axis=0).A1  # A1 = array 1D
        top_idx = mean_scores.argsort()[::-1][:8]
        top_terms = [(feature_names[i], round(mean_scores[i], 3)) for i in top_idx]
        print(f"\n{label_name}:")
        for term, score in top_terms:
            print(f"   {term:20s} {score}")

    return vectorizer, tfidf_matrix


def main():
    if len(sys.argv) != 2:
        print("Usage: python nlp_features.py emails_clean.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    # 1. Features additionnelles (numeriques simples)
    extra_features = build_additional_features(df)
    print("=" * 50)
    print("FEATURES ADDITIONNELLES (apercu)")
    print("=" * 50)
    print(extra_features.describe().round(1))

    # 2. TF-IDF
    vectorizer, tfidf_matrix = run_tfidf(df)

    # 3. Sauvegardes
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    extra_features["label"] = df["label"]
    extra_features.to_csv("features_preview.csv", index=False)

    print(f"\nVectorizer sauvegarde: tfidf_vectorizer.joblib")
    print(f"Features additionnelles sauvegardees: features_preview.csv")


if __name__ == "__main__":
    main()
