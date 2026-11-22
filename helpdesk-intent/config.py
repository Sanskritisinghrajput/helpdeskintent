"""Shared configuration for the Helpdesk Intent Classification Assistant."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SEED = 42

# Default helpdesk categories (the real label set is read from data/labels.json
# after you run prepare_data.py, so you can add your own categories).
DEFAULT_LABELS = ["account", "certification", "complaint", "process", "technical"]

# Where each category is routed in the (imaginary) institutional helpdesk.
DEPARTMENTS = {
    "account": {
        "title": "Account Services",
        "desc": "Login, password, OTP, profile and access issues.",
    },
    "certification": {
        "title": "Certification Cell",
        "desc": "Certificates, transcripts, verification and attestation.",
    },
    "complaint": {
        "title": "Grievance Redressal",
        "desc": "Complaints, delays, refunds and escalations.",
    },
    "process": {
        "title": "Academic & Admin Office",
        "desc": "Procedures, eligibility, deadlines, fees and appointments.",
    },
    "technical": {
        "title": "IT Support Desk",
        "desc": "Portal errors, uploads, payments gateway and app problems.",
    },
}
FALLBACK_DEPARTMENT = {
    "title": "Human Agent (Fallback)",
    "desc": "The model is not confident enough, so a person will pick this up.",
}

MODEL_NAMES = ["tfidf", "gru", "lstm", "bert"]
MODEL_TITLES = {
    "tfidf": "TF-IDF + Logistic Regression",
    "gru": "Bi-GRU (PyTorch)",
    "lstm": "Bi-LSTM (PyTorch)",
    "bert": "BERT (fine-tuned)",
}

DEFAULT_THRESHOLD = 0.60
MAX_QUERY_CHARS = 500
