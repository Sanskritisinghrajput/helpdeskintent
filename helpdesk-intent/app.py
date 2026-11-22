"""Flask web app for the Helpdesk Intent Classification Assistant.

    python app.py            ->  http://127.0.0.1:5000
"""
import json
import threading
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

from config import (DATA_DIR, DEFAULT_THRESHOLD, DEPARTMENTS, FALLBACK_DEPARTMENT, MAX_QUERY_CHARS,
                    MODEL_DIR, MODEL_NAMES, MODEL_TITLES)
from models_lib import load_model, model_exists

app = Flask(__name__)
_models, _lock = {}, threading.Lock()

EXAMPLES = [
    "I forgot my password and my account is locked",
    "How do I get a duplicate degree certificate?",
    "I was charged twice for the exam fee and nobody replies",
    "What is the last date to register for exams?",
    "The portal shows error 500 when I upload my documents",
    "What's the weather like in Mumbai today?",
    "Tell me a joke",
]


def labels():
    p = DATA_DIR / "labels.json"
    return json.loads(p.read_text()) if p.exists() else []


def results():
    p = MODEL_DIR / "results.json"
    return json.loads(p.read_text()) if p.exists() else {"models": {}}


def available():
    return [n for n in MODEL_NAMES if model_exists(n)]


def get_model(name):
    with _lock:
        if name not in _models:
            _models[name] = load_model(name, labels())
        return _models[name]


def default_threshold(name):
    r = results()["models"].get(name)
    return float(r["threshold"]) if r else DEFAULT_THRESHOLD


def classify(name, text, threshold):
    probs = get_model(name).predict_proba([text])[0]
    ls = labels()
    order = np.argsort(-probs)
    top = int(order[0])
    conf = float(probs[top])
    fallback = conf < threshold
    dept = FALLBACK_DEPARTMENT if fallback else DEPARTMENTS.get(ls[top], {"title": ls[top].title(), "desc": ""})
    return {
        "model": name,
        "model_title": MODEL_TITLES[name],
        "label": ls[top],
        "confidence": conf,
        "fallback": fallback,
        "threshold": threshold,
        "department": dept,
        "probabilities": [{"label": ls[int(i)], "p": float(probs[int(i)])} for i in order],
    }


def read_payload():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return None, None, ("Please type a question first.", 400)
    if len(text) > MAX_QUERY_CHARS:
        return None, None, (f"Query is too long (max {MAX_QUERY_CHARS} characters).", 400)
    thr = data.get("threshold")
    try:
        thr = None if thr is None else min(max(float(thr), 0.0), 1.0)
    except (TypeError, ValueError):
        return None, None, ("Invalid threshold.", 400)
    return data, (text, thr), None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    avail = available()
    return jsonify({
        "labels": labels(),
        "departments": DEPARTMENTS,
        "fallback_department": FALLBACK_DEPARTMENT,
        "models": [{"name": n, "title": MODEL_TITLES[n], "available": n in avail,
                    "threshold": default_threshold(n)} for n in MODEL_NAMES],
        "examples": EXAMPLES,
        "max_chars": MAX_QUERY_CHARS,
    })


@app.post("/api/predict")
def predict():
    data, parsed, err = read_payload()
    if err:
        return jsonify({"error": err[0]}), err[1]
    text, thr = parsed
    name = data.get("model") or (available() or [None])[0]
    if name not in available():
        return jsonify({"error": f"Model '{name}' is not trained yet. Run: python train.py --models {name}"}), 400
    thr = default_threshold(name) if thr is None else thr
    t0 = time.perf_counter()
    out = classify(name, text, thr)
    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return jsonify(out)


@app.post("/api/compare")
def compare():
    data, parsed, err = read_payload()
    if err:
        return jsonify({"error": err[0]}), err[1]
    text, thr = parsed
    out = []
    for name in available():
        t = default_threshold(name) if thr is None else thr
        t0 = time.perf_counter()
        r = classify(name, text, t)
        r["latency_ms"] = (time.perf_counter() - t0) * 1000
        out.append(r)
    return jsonify({"results": out})


@app.get("/api/metrics")
def metrics():
    return jsonify(results())


if __name__ == "__main__":
    print("Models available:", available() or "none - run: python train.py")
    app.run(host="127.0.0.1", port=5000, debug=False)
