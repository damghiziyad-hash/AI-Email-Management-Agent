"""
Phase 11 — Connexion Gmail API (MODE SECURISE)

Ce script se connecte a Gmail en LECTURE, fait passer chaque email par
l'Agent (Phase 10), puis AFFICHE la decision proposee sans jamais agir
automatiquement. C'est volontaire (cf. Phase 13 - Securite du roadmap):
  Email -> analyse -> PROPOSITION -> l'utilisateur valide manuellement

Une fois que tu as verifie sur plusieurs jours que les propositions sont
fiables, tu peux activer AUTO_APPLY = True pour laisser l'agent agir seul
sur les decisions ou la confiance/memoire est elevee (jamais sur les cas
INCERTAIN, qui demandent toujours une validation humaine).

Prerequis:
    pip install google-auth google-auth-oauthlib google-api-python-client --break-system-packages
    credentials.json present dans le meme dossier (voir guide de setup)

Usage:
    python gmail_connector.py
"""

import base64
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from email_agent import EmailAgent

# Scope en LECTURE + gestion des labels (necessaire pour "archiver"/"marquer important"
# plus tard) mais PAS de suppression definitive ni d'envoi.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

# ⚠️ Interrupteur de securite. Reste sur False jusqu'a validation manuelle
# de plusieurs jours de propositions correctes (Phase 14 du roadmap).
AUTO_APPLY = False


def get_gmail_service():
    """Authentification OAuth2, reutilise le token si deja valide."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_header(headers, name):
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def get_body_text(payload):
    """Extrait le texte brut du corps de l'email (gere les emails multipart)."""
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part["body"].get("data")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        # fallback: chercher plus profond (emails imbriques)
        for part in payload["parts"]:
            text = get_body_text(part)
            if text:
                return text
    else:
        data = payload.get("body", {}).get("data")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    return ""


def fetch_recent_emails(service, max_results=10):
    """Recupere les N emails les plus recents de l'Inbox."""
    results = service.users().messages().list(
        userId="me", labelIds=["INBOX"], maxResults=max_results
    ).execute()
    messages = results.get("messages", [])

    emails = []
    for msg_ref in messages:
        msg = service.users().messages().get(
            userId="me", id=msg_ref["id"], format="full"
        ).execute()
        headers = msg["payload"]["headers"]
        emails.append({
            "id": msg["id"],
            "sender": get_header(headers, "From"),
            "subject": get_header(headers, "Subject"),
            "body": get_body_text(msg["payload"])[:2000],  # on tronque, pas besoin de tout
        })
    return emails


def apply_action(service, email_id: str, decision: str):
    """
    Applique reellement l'action sur Gmail. N'est appele que si AUTO_APPLY=True
    ET que la decision n'est pas INCERTAIN.
    """
    if decision == "NORMAL":
        service.users().messages().modify(
            userId="me", id=email_id, body={"removeLabelIds": ["INBOX"]}
        ).execute()
    elif decision == "IMPORTANT":
        service.users().messages().modify(
            userId="me", id=email_id, body={"addLabelIds": ["IMPORTANT"]}
        ).execute()
    elif decision == "SPAM":
        service.users().messages().modify(
            userId="me", id=email_id, body={"addLabelIds": ["SPAM"], "removeLabelIds": ["INBOX"]}
        ).execute()


def main():
    print("Connexion a Gmail...")
    service = get_gmail_service()
    agent = EmailAgent()

    print("Recuperation des emails recents...")
    emails = fetch_recent_emails(service, max_results=10)

    print(f"\n{len(emails)} emails recuperes. Analyse en cours...\n")
    print(f"Mode: {'ACTION AUTOMATIQUE' if AUTO_APPLY else 'PROPOSITION SEULEMENT (securise)'}\n")

    for email in emails:
        result = agent.process_email(email["sender"], email["subject"], email["body"])

        print(f"De: {email['sender']}")
        print(f"Objet: {email['subject']}")
        print(f"-> Proposition: {result['decision']} ({result['action']})")
        print(f"-> Raison: {result['reason']}")

        if AUTO_APPLY and result["decision"] != "INCERTAIN":
            apply_action(service, email["id"], result["decision"])
            print("-> ACTION APPLIQUEE AUTOMATIQUEMENT")
        else:
            print("-> Aucune action appliquee (mode securise / cas incertain)")
        print()


if __name__ == "__main__":
    main()
