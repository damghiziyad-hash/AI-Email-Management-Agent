# 📧 AI Email Management Agent

Un agent qui classe automatiquement les emails Gmail en **IMPORTANT / NORMAL / SPAM**, en combinant NLP, Machine Learning, un moteur de décision basé sur la confiance, et une mémoire utilisateur qui apprend des corrections manuelles.

Projet construit étape par étape pour apprendre concrètement le pipeline complet d'un système ML en production : de la donnée brute jusqu'à l'intégration avec une vraie API (Gmail).

---

## 🎯 Objectif

```
📧 Nouvel email
      ↓
🧠 Analyse IA (TF-IDF + ML)
      ↓
┌──────────────┬──────────────┬──────────────┐
│ 🔴 Important │ 🟡 Normal    │ ⚫ Spam      │
└──────────────┴──────────────┴──────────────┘
      ↓               ↓              ↓
   Prioritaire      Archive        Spam
```

L'agent ne se contente pas d'un modèle ML brut : il passe par un **moteur de décision** qui refuse de trancher quand la confiance est insuffisante (`INCERTAIN`), et une **mémoire utilisateur** qui corrige le modèle au fil du temps en fonction des choix réels de l'utilisateur.

---

## 🏗️ Architecture

```
Gmail API
    │
    ▼
Email Agent
    │
    ├── Classifier (TF-IDF + Naive Bayes/Logistic Regression)
    ├── Decision Engine (seuils de confiance, asymétrie du coût des erreurs)
    └── User Memory (SQLite — override par expéditeur)
    │
    ▼
Action: Inbox / Archive / Spam / Demande de confirmation
```

---

## 📂 Structure du projet

| Fichier | Rôle |
|---|---|
| `clean_emails.py` | Nettoyage du dataset (HTML, doublons, normalisation texte) |
| `eda_emails.py` | Analyse exploratoire (distribution, mots fréquents, longueur) |
| `nlp_features.py` | Transformation du texte en features numériques (TF-IDF) |
| `train_models.py` | Entraînement et comparaison de plusieurs modèles ML |
| `evaluate_model.py` | Évaluation détaillée (precision/recall/F1, matrice de confusion) |
| `decision_engine.py` | Couche de décision basée sur les probabilités et seuils de confiance |
| `user_memory.py` | Mémoire SQLite qui apprend des corrections de l'utilisateur |
| `email_agent.py` | Assemble classifier + decision engine + mémoire en un agent complet |
| `gmail_connector.py` | Connexion à l'API Gmail en mode sécurisé (lecture + proposition uniquement) |

---

## 🚀 Installation

```bash
git clone <ton-repo>
cd ai-email-management-agent
pip install -r requirements.txt
```

### Connexion Gmail (optionnel)

1. Crée un projet sur [Google Cloud Console](https://console.cloud.google.com), active l'API Gmail
2. Configure l'écran de consentement OAuth (mode Test) et ajoute-toi comme test user
3. Crée un client OAuth (type "Application de bureau"), télécharge `credentials.json`
4. Place `credentials.json` à la racine du projet (⚠️ ne jamais le commit, voir `.gitignore`)
5. Lance :

```bash
python gmail_connector.py
```

Un flux d'autorisation s'ouvre dans le navigateur. Le script tourne par défaut en **mode sécurisé** (`AUTO_APPLY = False`) : il propose une classification pour chaque email mais n'applique **aucune action automatique**.

---

## 📊 Résultats actuels

> ⚠️ Le dataset actuel est un petit jeu d'exemples synthétiques (25 emails) construit pour valider le pipeline de bout en bout. Les métriques ci-dessous ne sont donc pas représentatives d'une performance en conditions réelles — l'étape suivante du projet est l'entraînement sur un dataset public à plus grande échelle (Enron, SpamAssassin).

- Pipeline validé de bout en bout : nettoyage → NLP → ML → évaluation → décision → mémoire → agent → Gmail
- Connexion Gmail réelle testée avec succès en mode lecture/proposition

---

## 🔐 Sécurité

- Mode "proposition seulement" par défaut — aucune action Gmail automatique tant que non activée explicitement
- Scope OAuth limité à `gmail.modify` (pas de suppression définitive, pas d'envoi d'email)
- `credentials.json`, `token.json` et la base mémoire (`user_memory.db`) sont exclus du repo (`.gitignore`)

---

## 🗺️ Roadmap

- [x] Définition du problème
- [x] Dataset (starter)
- [x] Data cleaning
- [x] EDA
- [x] NLP (TF-IDF)
- [x] Premiers modèles ML
- [x] Évaluation (precision/recall/F1)
- [x] Decision engine
- [x] Mémoire utilisateur
- [x] Agent complet
- [x] Connexion Gmail (mode sécurisé)
- [ ] Dashboard (Streamlit)
- [ ] Sécurité renforcée / audit
- [ ] Tests en conditions réelles à grande échelle
- [ ] Amélioration itérative
- [ ] Déploiement

---

## 🛠️ Stack technique

Python · pandas · scikit-learn · TF-IDF · SQLite · Gmail API (OAuth2)
