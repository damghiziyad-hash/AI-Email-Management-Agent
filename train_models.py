"""
Phase 6 — Premier modele ML

Usage:
    python train_models.py emails_clean.csv

Entraine 3 modeles (Logistic Regression, Naive Bayes, Random Forest)
via un Pipeline sklearn (TF-IDF + classifieur), pour eviter toute
fuite de donnees: le TF-IDF est fit UNIQUEMENT sur le train set.

Sauvegarde le meilleur pipeline (vectorizer + modele ensemble) pret
a etre reutilise en Phase 7 (evaluation approfondie) et Phase 11 (Gmail).
"""

import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

LABELS = {0: "NORMAL", 1: "IMPORTANT", 2: "SPAM"}

FRENCH_STOPWORDS = [
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "a", "au", "aux",
    "ce", "cet", "cette", "ces", "pour", "dans", "que", "qui", "est", "sont", "sur",
    "avec", "par", "se", "ne", "pas", "vous", "nous", "votre", "vos", "notre", "nos",
    "il", "elle", "ils", "elles", "je", "tu", "on", "son", "sa", "ses", "leur",
    "leurs", "plus", "moins", "tres", "bien", "tout", "tous", "toute", "d", "l",
    "n", "c", "j", "s", "qu", "etre", "avoir",
]


def build_pipeline(classifier):
    """TF-IDF + classifieur, assembles dans un seul objet reutilisable."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=FRENCH_STOPWORDS,
        )),
        ("clf", classifier),
    ])


def main():
    if len(sys.argv) != 2:
        print("Usage: python train_models.py emails_clean.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    X = (df["subject_clean"].fillna("") + " " + df["body_clean"].fillna("")).values
    y = df["label"].values

    # Split AVANT tout fit -> pas de fuite de donnees
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} emails | Test: {len(X_test)} emails")
    print(f"(dataset volontairement petit pour l'instant, cf. note Phase 6)\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []
    trained_pipelines = {}

    for name, clf in models.items():
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))

        results.append({"Modele": name, "Train Accuracy": train_acc, "Test Accuracy": test_acc})
        trained_pipelines[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
    print("=" * 55)
    print("COMPARAISON DES MODELES")
    print("=" * 55)
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Modele"]
    best_pipeline = trained_pipelines[best_name]
    print(f"\nMeilleur modele (sur ce petit test set): {best_name}")

    joblib.dump(best_pipeline, "best_model_pipeline.joblib")
    print(f"Pipeline complet (TF-IDF + modele) sauvegarde: best_model_pipeline.joblib")

    # On sauvegarde aussi les predictions detaillees pour la Phase 7 (evaluation)
    predictions_detail = pd.DataFrame({
        "text": X_test,
        "true_label": [LABELS[v] for v in y_test],
        "predicted_label": [LABELS[v] for v in best_pipeline.predict(X_test)],
    })
    predictions_detail.to_csv("test_predictions.csv", index=False)
    print("Predictions detaillees sauvegardees: test_predictions.csv (pour Phase 7)")


if __name__ == "__main__":
    main()
