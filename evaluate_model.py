"""
Phase 7 — Evaluation

Usage:
    python evaluate_model.py emails_clean.csv best_model_pipeline.joblib

Va au-dela de l'accuracy: precision/recall/F1 par classe, matrice de
confusion, et un focus specifique sur l'erreur la plus grave du projet:
IMPORTANT classe comme SPAM (email critique perdu dans le dossier spam).
"""

import sys

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

LABELS = {0: "NORMAL", 1: "IMPORTANT", 2: "SPAM"}
LABEL_NAMES = list(LABELS.values())


def reproduce_test_split(df):
    """Recree exactement le meme split que la Phase 6 (meme random_state)."""
    X = (df["subject_clean"].fillna("") + " " + df["body_clean"].fillna("")).values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_test, y_test


def check_critical_errors(y_test, y_pred):
    """Verifie specifiquement les erreurs IMPORTANT -> SPAM (les plus graves)."""
    critical_errors = 0
    for true_val, pred_val in zip(y_test, y_pred):
        if LABELS[true_val] == "IMPORTANT" and LABELS[pred_val] == "SPAM":
            critical_errors += 1

    print("=" * 55)
    print("VERIFICATION DES ERREURS CRITIQUES (IMPORTANT -> SPAM)")
    print("=" * 55)
    if critical_errors == 0:
        print("Aucune erreur critique detectee sur ce test set.")
    else:
        print(f"ATTENTION: {critical_errors} email(s) IMPORTANT classe(s) SPAM.")
        print("C'est l'erreur la plus grave possible pour ce projet:")
        print("un email potentiellement urgent finirait invisible dans le spam.")


def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_model.py emails_clean.csv best_model_pipeline.joblib")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    pipeline = joblib.load(sys.argv[2])

    X_test, y_test = reproduce_test_split(df)
    y_pred = pipeline.predict(X_test)

    print("=" * 55)
    print("RAPPORT DE CLASSIFICATION (precision / recall / F1)")
    print("=" * 55)
    report = classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES,
        labels=list(LABELS.keys()),
        zero_division=0,
    )
    print(report)

    print("Rappel des definitions:")
    print("  Precision = parmi ce que le modele a predit X, combien etait vraiment X")
    print("  Recall    = parmi les vrais X, combien le modele en a retrouve")
    print("  -> Pour SPAM: on veut une PRECISION elevee (peu de faux positifs)")
    print("  -> Pour IMPORTANT: on veut un RECALL eleve (n'en rater aucun)")
    print()

    check_critical_errors(y_test, y_pred)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=list(LABELS.keys()))
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    print("\nMatrice de confusion sauvegardee: confusion_matrix.png")


if __name__ == "__main__":
    main()
