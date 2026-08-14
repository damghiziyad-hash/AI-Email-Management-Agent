"""
Phase 10 — L'Agent

Assemble tout ce qui a ete construit dans les phases precedentes:
  - Phase 6: modele ML (pipeline TF-IDF + classifieur)
  - Phase 8: moteur de decision (probabilites -> decision + seuils)
  - Phase 9: memoire utilisateur (override par expediteur)

L'agent expose une seule methode: process_email(sender, subject, body)
-> qui retourne la decision finale et l'action a effectuer.

Usage (demo):
    python email_agent.py
"""

import re

import joblib

from user_memory import init_db, apply_memory_to_decision, record_user_decision

LABELS = {0: "NORMAL", 1: "IMPORTANT", 2: "SPAM"}

CONFIDENCE_THRESHOLD = 0.60
SPAM_CONFIDENCE_THRESHOLD = 0.80

ACTIONS = {
    "IMPORTANT": "garder en Inbox + notifier",
    "NORMAL": "archiver",
    "SPAM": "deplacer vers Spam",
    "INCERTAIN": "demander confirmation a l'utilisateur",
}


def normalize_text(text: str) -> str:
    """Meme logique que la Phase 3, simplifiee pour un email unique en entree live."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email ", text)
    text = re.sub(r"[^a-z0-9àâäéèêëîïôöùûüç\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EmailAgent:
    def __init__(self, model_path="best_model_pipeline.joblib", db_path="user_memory.db"):
        self.pipeline = joblib.load(model_path)
        self.memory_conn = init_db(db_path)

    def _ml_decision(self, subject: str, body: str) -> dict:
        text = normalize_text(subject) + " " + normalize_text(body)
        probas = self.pipeline.predict_proba([text])[0]
        prob_dict = {LABELS[i]: p for i, p in enumerate(probas)}

        best_label = max(prob_dict, key=prob_dict.get)
        best_proba = prob_dict[best_label]

        if best_label == "SPAM":
            if best_proba >= SPAM_CONFIDENCE_THRESHOLD:
                return {"decision": "SPAM", "auto": True,
                        "reason": f"confiance SPAM {best_proba:.0%}", "probabilities": prob_dict}
            return {"decision": "INCERTAIN", "auto": False,
                    "reason": f"confiance SPAM {best_proba:.0%} < seuil {SPAM_CONFIDENCE_THRESHOLD:.0%}",
                    "probabilities": prob_dict}

        if best_proba >= CONFIDENCE_THRESHOLD:
            return {"decision": best_label, "auto": True,
                    "reason": f"confiance {best_proba:.0%}", "probabilities": prob_dict}

        return {"decision": "INCERTAIN", "auto": False,
                "reason": f"confiance insuffisante (max {best_proba:.0%})", "probabilities": prob_dict}

    def process_email(self, sender: str, subject: str, body: str) -> dict:
        """Point d'entree principal de l'agent."""
        ml_result = self._ml_decision(subject, body)
        final = apply_memory_to_decision(self.memory_conn, sender, ml_result)
        final["action"] = ACTIONS[final["decision"]]
        final["sender"] = sender
        final["subject"] = subject
        return final

    def user_correction(self, sender: str, correct_label_name: str):
        """A appeler quand l'utilisateur corrige manuellement une decision."""
        label_id = {v: k for k, v in LABELS.items()}[correct_label_name]
        record_user_decision(self.memory_conn, sender, label_id)


def print_result(result: dict):
    print(f"De: {result['sender']}")
    print(f"Objet: {result['subject']}")
    print(f"-> Decision: {result['decision']}  (source: {result['source']})")
    print(f"-> Action: {result['action']}")
    print(f"-> Raison: {result['reason']}")
    print()


def demo():
    agent = EmailAgent()

    print("=" * 65)
    print("EMAIL 1 — expediteur inconnu, le ML decide seul")
    print("=" * 65)
    r1 = agent.process_email(
        sender="rh@monentreprise.com",
        subject="Convocation entretien annuel",
        body="Votre entretien est fixe au 20 fevrier a 10h en salle B.",
    )
    print_result(r1)

    print("=" * 65)
    print("EMAIL 2 — nouveau spam, ML incertain -> demande confirmation")
    print("=" * 65)
    r2 = agent.process_email(
        sender="deals@super-promo-xyz.com",
        subject="Offre exceptionnelle juste pour vous",
        body="Cliquez ici pour decouvrir notre offre limitee dans le temps.",
    )
    print_result(r2)

    print("L'utilisateur corrige manuellement -> SPAM (2 fois, pour atteindre le seuil memoire)")
    agent.user_correction("deals@super-promo-xyz.com", "SPAM")
    agent.user_correction("deals@super-promo-xyz.com", "SPAM")
    print()

    print("=" * 65)
    print("EMAIL 3 — meme expediteur, la memoire prend le relais")
    print("=" * 65)
    r3 = agent.process_email(
        sender="deals@super-promo-xyz.com",
        subject="Derniere chance: -70% aujourd'hui seulement",
        body="Ne manquez pas cette offre qui expire ce soir.",
    )
    print_result(r3)


if __name__ == "__main__":
    demo()
