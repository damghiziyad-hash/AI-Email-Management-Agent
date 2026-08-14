"""
Phase 3 — Data Cleaning
Script de nettoyage reutilisable pour le dataset d'emails.

Usage:
    python clean_emails.py input.csv output_clean.csv

Colonnes attendues en entree: email_id, sender, subject, body, label
"""

import sys
import re
import pandas as pd


def strip_html(text: str) -> str:
    """Supprime les balises HTML residuelles."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # balises HTML
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;", " ", text)  # entites HTML courantes
    return text


def normalize_text(text: str) -> str:
    """Normalise le texte: minuscules, espaces, caracteres speciaux."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)     # liens -> token URL
    text = re.sub(r"\S+@\S+\.\S+", " EMAIL ", text)        # adresses email dans le corps
    text = re.sub(r"[^a-z0-9àâäéèêëîïôöùûüç\s]", " ", text)  # garde lettres/chiffres/accents FR
    text = re.sub(r"\s+", " ", text).strip()               # espaces multiples
    return text


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)

    # 1. Supprimer les lignes avec colonnes essentielles manquantes
    df = df.dropna(subset=["sender", "subject", "body", "label"])
    n_after_na = len(df)

    # 2. Nettoyer HTML puis normaliser subject/body
    df["subject_clean"] = df["subject"].apply(strip_html).apply(normalize_text)
    df["body_clean"] = df["body"].apply(strip_html).apply(normalize_text)

    # 3. Supprimer les emails devenus vides ou trop courts (inutilisables)
    df = df[(df["body_clean"].str.len() >= 5)]
    n_after_short = len(df)

    # 4. Nettoyer le sender (minuscule, espaces)
    df["sender"] = df["sender"].astype(str).str.strip().str.lower()

    # 5. Supprimer les doublons (sur sender + subject_clean + body_clean)
    df = df.drop_duplicates(subset=["sender", "subject_clean", "body_clean"])
    n_after_dupes = len(df)

    # 6. Longueur du texte (utile pour l'EDA en Phase 4)
    df["text_length"] = (df["subject_clean"] + " " + df["body_clean"]).str.len()

    # 7. Reset index / email_id propre
    df = df.reset_index(drop=True)
    df["email_id"] = range(1, len(df) + 1)

    print("=== Rapport de nettoyage ===")
    print(f"Emails au depart          : {n_start}")
    print(f"Apres suppression NA      : {n_after_na}  (-{n_start - n_after_na})")
    print(f"Apres suppression courts  : {n_after_short}  (-{n_after_na - n_after_short})")
    print(f"Apres suppression doublons: {n_after_dupes}  (-{n_after_short - n_after_dupes})")
    print(f"Total final               : {len(df)}")
    print()
    print("Distribution des classes (finale):")
    print(df["label"].map({0: "NORMAL", 1: "IMPORTANT", 2: "SPAM"}).value_counts())

    return df


def main():
    if len(sys.argv) != 3:
        print("Usage: python clean_emails.py input.csv output_clean.csv")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(input_path)
    df_clean = clean_dataset(df)

    cols = ["email_id", "sender", "subject_clean", "body_clean", "text_length", "label"]
    df_clean[cols].to_csv(output_path, index=False)
    print(f"\nFichier nettoye sauvegarde: {output_path}")


if __name__ == "__main__":
    main()
