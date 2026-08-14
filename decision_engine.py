"""
Phase 8 — Decision Engine

Usage:
    python decision_engine.py emails_clean.csv best_model_pipeline.joblib

Le modele ML ne decide jamais seul. On ajoute une couche de decision qui:
  1. Regarde la probabilite de chaque classe
  2. Si une classe est tres confiante -> decision automatique
  3. Sinon -> "incertain", on demande confirmation a l'utilisateur
  4. Regle de securite specifique: ne JAMAIS classer SPAM automatiquement
     sauf si la confiance est tres elevee (asymetrie du cout des erreurs,
     cf. Phase 7: rater un IMPORTANT est bien plus grave que garder un
     spam par erreur dans l'inbox).
"""

import sys

import joblib
import pandas as pd

LABELS = {0: "NORMAL", 1: "IMPORTANT", 2: "SPAM"}

# Seuils de decision (a ajuster avec plus de donnees / tests reels)
CONFIDENCE_THRESHOLD = 0.60      # confiance minimale pour decider seul
SPAM_CONFIDENCE_THRESHOLD = 0.80  # seuil plus strict specifiquement pour SPAM


def decide(probabilities: dict) -> dict:
    """
    probabilities: {"NORMAL": 0.1, "IMPORTANT": 0.85, "SPAM": 0.05}
    Retourne la decision finale + le niveau de confiance + la raison.
    """
    best_label = max(probabilities, key=probabilities.get)
    best_proba = probabilities[best_label]

    # Regle de securite: SPAM demande un seuil plus eleve que les autres
    if best_label == "SPAM":
        if best_proba >= SPAM_CONFIDENCE_THRESHOLD:
            return {"decision": "SPAM", "auto": True,
                    "reason": f"confiance SPAM elevee ({best_proba:.0%})"}
        else:
            return {"decision": "INCERTAIN", "auto": False,
                    "reason": f"confiance SPAM insuffisante ({best_proba:.0%} < {SPAM_CONFIDENCE_THRESHOLD:.0%}) "
                               f"-> on prefere demander confirmation plutot que de risquer un faux positif"}

    # Pour NORMAL / IMPORTANT: seuil standard
    if best_proba >= CONFIDENCE_THRESHOLD:
        return {"decision": best_label, "auto": True,
                "reason": f"confiance suffisante ({best_proba:.0%})"}

    return {"decision": "INCERTAIN", "auto": False,
            "reason": f"aucune classe assez confiante (max: {best_proba:.0%})"}


def action_for(decision: str) -> str:
    return {
        "IMPORTANT": "garder en Inbox + notifier",
        "NORMAL": "archiver",
        "SPAM": "deplacer vers Spam",
        "INCERTAIN": "demander confirmation a l'utilisateur",
    }[decision]


def main():
    if len(sys.argv) != 3:
        print("Usage: python decision_engine.py emails_clean.csv best_model_pipeline.joblib")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    pipeline = joblib.load(sys.argv[2])

    full_text = (df["subject_clean"].fillna("") + " " + df["body_clean"].fillna("")).values
    probas = pipeline.predict_proba(full_text)

    print("=" * 70)
    print(f"Seuils actifs: standard={CONFIDENCE_THRESHOLD:.0%} | SPAM={SPAM_CONFIDENCE_THRESHOLD:.0%}")
    print("=" * 70)

    rows = []
    for i, row_probas in enumerate(probas):
        prob_dict = {LABELS[j]: p for j, p in enumerate(row_probas)}
        result = decide(prob_dict)
        rows.append({
            "email_id": df.iloc[i]["email_id"],
            "true_label": LABELS[df.iloc[i]["label"]],
            "P(NORMAL)": round(prob_dict["NORMAL"], 2),
            "P(IMPORTANT)": round(prob_dict["IMPORTANT"], 2),
            "P(SPAM)": round(prob_dict["SPAM"], 2),
            "decision": result["decision"],
            "action": action_for(result["decision"]),
            "raison": result["reason"],
        })

    results_df = pd.DataFrame(rows)
    print(results_df[["email_id", "true_label", "P(NORMAL)", "P(IMPORTANT)", "P(SPAM)", "decision"]].to_string(index=False))

    n_incertain = (results_df["decision"] == "INCERTAIN").sum()
    n_auto = len(results_df) - n_incertain
    print(f"\nDecisions automatiques: {n_auto}/{len(results_df)}")
    print(f"Demandes de confirmation: {n_incertain}/{len(results_df)}")

    results_df.to_csv("decision_engine_output.csv", index=False)
    print("\nDetail complet sauvegarde: decision_engine_output.csv")


if __name__ == "__main__":
    main()
