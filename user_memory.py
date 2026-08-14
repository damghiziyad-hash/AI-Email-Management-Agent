"""
Phase 9 — Memoire utilisateur

Stocke, par expediteur, l'historique des decisions de l'utilisateur
(corrections manuelles). Cette memoire peut ensuite surclasser (override)
la prediction du modele ML quand elle est suffisamment fiable.

Usage (demo):
    python user_memory.py
"""

import sqlite3
from datetime import datetime

DB_PATH = "user_memory.db"

# Seuil: nombre minimum de decisions coherentes de l'utilisateur
# avant que la memoire ait le droit d'overrider le modele ML
OVERRIDE_THRESHOLD = 2
LABELS = {0: "NORMAL", 1: "IMPORTANT", 2: "SPAM"}


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sender_memory (
            sender TEXT PRIMARY KEY,
            normal_count INTEGER DEFAULT 0,
            important_count INTEGER DEFAULT 0,
            spam_count INTEGER DEFAULT 0,
            last_updated TEXT
        )
    """)
    conn.commit()
    return conn


def record_user_decision(conn, sender: str, label: int):
    """Appele quand l'utilisateur confirme/corrige une decision manuellement."""
    sender = sender.strip().lower()
    label_name = LABELS[label]
    column = f"{label_name.lower()}_count"

    conn.execute(f"""
        INSERT INTO sender_memory (sender, {column}, last_updated)
        VALUES (?, 1, ?)
        ON CONFLICT(sender) DO UPDATE SET
            {column} = {column} + 1,
            last_updated = excluded.last_updated
    """, (sender, datetime.now().isoformat()))
    conn.commit()
    print(f"[memoire] {sender} -> {label_name} enregistre")


def get_sender_memory(conn, sender: str):
    sender = sender.strip().lower()
    row = conn.execute(
        "SELECT normal_count, important_count, spam_count FROM sender_memory WHERE sender = ?",
        (sender,)
    ).fetchone()
    if row is None:
        return None
    return {"NORMAL": row[0], "IMPORTANT": row[1], "SPAM": row[2]}


def memory_override(conn, sender: str):
    """
    Regarde si la memoire est assez forte pour overrider le modele ML.
    Retourne le label a forcer, ou None si pas assez de donnees / pas de consensus clair.
    """
    counts = get_sender_memory(conn, sender)
    if counts is None:
        return None

    total = sum(counts.values())
    if total < OVERRIDE_THRESHOLD:
        return None  # pas assez d'historique

    best_label = max(counts, key=counts.get)
    best_count = counts[best_label]

    # On exige que la classe dominante represente une nette majorite (>= 70%)
    if best_count / total >= 0.7:
        return best_label
    return None  # historique trop mitige, on laisse le modele ML decider


def apply_memory_to_decision(conn, sender: str, ml_decision: dict) -> dict:
    """
    Combine memoire utilisateur + decision du moteur (Phase 8).
    La memoire, quand elle est fiable, a priorite sur le ML.
    """
    override = memory_override(conn, sender)
    if override is not None:
        return {
            "decision": override,
            "auto": True,
            "reason": f"override memoire utilisateur (historique fiable pour {sender})",
            "source": "memory",
        }

    ml_decision["source"] = "ml_model"
    return ml_decision


def demo():
    conn = init_db()

    print("=" * 60)
    print("SIMULATION: l'utilisateur corrige des decisions a la main")
    print("=" * 60)

    # L'utilisateur deplace 2x une newsletter vers Spam
    record_user_decision(conn, "newsletter@promo-deals.com", label=2)  # SPAM
    record_user_decision(conn, "newsletter@promo-deals.com", label=2)  # SPAM

    # L'utilisateur marque 2x un recruteur comme important
    record_user_decision(conn, "recruiter@techcorp.com", label=1)  # IMPORTANT
    record_user_decision(conn, "recruiter@techcorp.com", label=1)  # IMPORTANT

    print()
    print("=" * 60)
    print("NOUVEL EMAIL RECU: newsletter@promo-deals.com")
    print("=" * 60)
    # Supposons que le modele ML hesite (cas frequent vu Phase 7/8)
    ml_decision = {"decision": "INCERTAIN", "auto": False, "reason": "confiance ML insuffisante"}
    final = apply_memory_to_decision(conn, "newsletter@promo-deals.com", ml_decision)
    print(f"Decision ML seule       : {ml_decision['decision']}")
    print(f"Decision finale (avec memoire): {final['decision']}  (source: {final['source']})")
    print(f"Raison: {final['reason']}")

    print()
    print("=" * 60)
    print("NOUVEL EMAIL RECU: recruiter@techcorp.com")
    print("=" * 60)
    ml_decision2 = {"decision": "NORMAL", "auto": True, "reason": "confiance ML 65%"}
    final2 = apply_memory_to_decision(conn, "recruiter@techcorp.com", ml_decision2)
    print(f"Decision ML seule       : {ml_decision2['decision']}")
    print(f"Decision finale (avec memoire): {final2['decision']}  (source: {final2['source']})")
    print(f"Raison: {final2['reason']}")
    print()
    print("-> Meme si le ML dit NORMAL, la memoire (2x IMPORTANT confirme par l'utilisateur)")
    print("   prend le dessus. C'est l'interet principal de cette phase.")

    print()
    print("=" * 60)
    print("NOUVEL EMAIL RECU: inconnu@jamais-vu.com (pas d'historique)")
    print("=" * 60)
    ml_decision3 = {"decision": "SPAM", "auto": True, "reason": "confiance ML 82%"}
    final3 = apply_memory_to_decision(conn, "inconnu@jamais-vu.com", ml_decision3)
    print(f"Decision finale: {final3['decision']}  (source: {final3['source']})")
    print("-> Pas de memoire pour cet expediteur, le modele ML garde la main.")

    conn.close()


if __name__ == "__main__":
    demo()
