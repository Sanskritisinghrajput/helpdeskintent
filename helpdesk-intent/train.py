"""Train and compare TF-IDF, Bi-GRU, Bi-LSTM and BERT, with confidence-based fallback.

python train.py                          # all four models
python train.py --models tfidf gru       # only some
python train.py --bert-model distilbert-base-uncased --bert-epochs 3
python train.py --models bert --bert-model prajjwal1/bert-tiny   # tiny BERT, fast on CPU

Results are merged into models/results.json (the website reads this file).
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import DATA_DIR, MODEL_DIR, MODEL_NAMES, MODEL_TITLES, SEED
from models_lib import BertModel, RNNModel, TfidfModel, set_seed

THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 1.0, 0.01)]


def load_data():
    need = ["train", "val", "test", "val_oos", "test_oos"]
    missing = [n for n in need if not (DATA_DIR / f"{n}.csv").exists()]
    if missing:
        raise SystemExit(f"Missing {missing}. Run:  python prepare_data.py")
    labels = json.loads((DATA_DIR / "labels.json").read_text())
    l2i = {l: i for i, l in enumerate(labels)}
    d = {n: pd.read_csv(DATA_DIR / f"{n}.csv").dropna() for n in need}
    for n in ("train", "val", "test"):
        d[n]["y"] = d[n]["label"].map(l2i)
    return labels, d


def fallback_sweep(conf_in, correct_in, conf_oos):
    """For each threshold: how much in-scope traffic is auto-handled correctly,
    how much accepted traffic is right, and how much out-of-scope is sent to a human."""
    rows = []
    for t in THRESHOLDS:
        accepted = conf_in >= t
        handled_ok = float((accepted & correct_in).mean())
        rows.append({
            "threshold": t,
            "coverage": float(accepted.mean()),                                   # in-scope auto-answered
            "accepted_accuracy": float(correct_in[accepted].mean()) if accepted.any() else 0.0,
            "in_scope_handled_ok": handled_ok,
            "oos_rejection": float((conf_oos < t).mean()) if len(conf_oos) else 0.0,
        })
    return rows


def pick_threshold(sweep):
    """Balance: auto-handle in-scope correctly AND reject out-of-scope. Chosen on VALIDATION data."""
    best = max(sweep, key=lambda r: (0.5 * (r["in_scope_handled_ok"] + r["oos_rejection"]), r["threshold"]))
    return best["threshold"]


def evaluate(model, labels, d):
    test_texts, y_test = d["test"]["text"].tolist(), d["test"]["y"].to_numpy()
    p_test = model.predict_proba(test_texts)
    pred = p_test.argmax(1)

    # threshold chosen on validation in-scope + validation OOS
    p_val = model.predict_proba(d["val"]["text"].tolist())
    p_val_oos = model.predict_proba(d["val_oos"]["text"].tolist())
    val_sweep = fallback_sweep(p_val.max(1), p_val.argmax(1) == d["val"]["y"].to_numpy(), p_val_oos.max(1))
    thr = pick_threshold(val_sweep)

    # report on TEST
    p_test_oos = model.predict_proba(d["test_oos"]["text"].tolist())
    test_sweep = fallback_sweep(p_test.max(1), pred == y_test, p_test_oos.max(1))
    at_thr = min(test_sweep, key=lambda r: abs(r["threshold"] - thr))

    sample = test_texts[:50]
    t0 = time.perf_counter()
    for s in sample:
        model.predict_proba([s])
    latency_ms = (time.perf_counter() - t0) / len(sample) * 1000

    rep = classification_report(y_test, pred, target_names=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "per_class": {l: {k: float(rep[l][k]) for k in ("precision", "recall", "f1-score", "support")} for l in labels},
        "confusion_matrix": confusion_matrix(y_test, pred, labels=list(range(len(labels)))).tolist(),
        "mean_confidence_correct": float(p_test.max(1)[pred == y_test].mean()),
        "mean_confidence_wrong": float(p_test.max(1)[pred != y_test].mean()) if (pred != y_test).any() else None,
        "mean_confidence_oos": float(p_test_oos.max(1).mean()),
        "latency_ms": latency_ms,
        "threshold": thr,
        "at_threshold": at_thr,
        "sweep": test_sweep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--rnn-epochs", type=int, default=25)
    ap.add_argument("--bert-epochs", type=int, default=8)
    ap.add_argument("--bert-model", default="bert-base-uncased")
    ap.add_argument("--bert-batch", type=int, default=32)
    ap.add_argument("--bert-lr", type=float, default=5e-5)
    ap.add_argument("--limit-train", type=int, default=0, help="debug: use only N training rows")
    args = ap.parse_args()

    set_seed(SEED)
    labels, d = load_data()
    if args.limit_train:
        d["train"] = d["train"].sample(args.limit_train, random_state=SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    print(f"labels={labels}\ntrain={len(d['train'])} val={len(d['val'])} test={len(d['test'])} "
          f"oos_val={len(d['val_oos'])} oos_test={len(d['test_oos'])}\n")

    res_path = MODEL_DIR / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {"models": {}}
    results.update({
        "labels": labels,
        "dataset": {"train": len(d["train"]), "val": len(d["val"]), "test": len(d["test"]),
                    "oos_val": len(d["val_oos"]), "oos_test": len(d["test_oos"])},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    })

    tr_x, tr_y = d["train"]["text"].tolist(), d["train"]["y"].tolist()
    va_x, va_y = d["val"]["text"].tolist(), d["val"]["y"].tolist()

    for name in args.models:
        print(f"=== {name.upper()} ({MODEL_TITLES[name]}) ===")
        t0 = time.time()
        try:
            if name == "tfidf":
                model = TfidfModel(labels).fit(tr_x, tr_y)
            elif name in ("gru", "lstm"):
                model = RNNModel(labels, cell=name).fit(tr_x, tr_y, va_x, va_y, epochs=args.rnn_epochs)
            else:
                model = BertModel(labels).fit(tr_x, tr_y, va_x, va_y, epochs=args.bert_epochs,
                                              batch_size=args.bert_batch, lr=args.bert_lr,
                                              pretrained=args.bert_model)
        except ImportError as e:
            print(f"  skipped {name}: {e}  (pip install -r requirements.txt)\n")
            continue
        train_s = time.time() - t0
        model.save(MODEL_DIR)
        r = evaluate(model, labels, d)
        r.update({"title": MODEL_TITLES[name], "train_seconds": train_s})
        if name == "bert":
            r["pretrained"] = args.bert_model
        results["models"][name] = r
        a = r["at_threshold"]
        print(f"  test accuracy {r['accuracy']:.4f} | macro-F1 {r['macro_f1']:.4f} | train {train_s:.1f}s | {r['latency_ms']:.1f} ms/query")
        print(f"  fallback @ {r['threshold']:.2f}: auto-handled {a['coverage']:.1%}, accepted accuracy {a['accepted_accuracy']:.1%}, "
              f"out-of-scope sent to human {a['oos_rejection']:.1%}\n")
        res_path.write_text(json.dumps(results, indent=1))

    # leaderboard
    print("MODEL      ACC     F1      THR   ACC@accepted  OOS-rejected  ms/query")
    for n, r in results["models"].items():
        a = r["at_threshold"]
        print(f"{n:9s} {r['accuracy']:.4f}  {r['macro_f1']:.4f}  {r['threshold']:.2f}  "
              f"{a['accepted_accuracy']:.4f}        {a['oos_rejection']:.4f}        {r['latency_ms']:.1f}")
    print(f"\nSaved {res_path}")


if __name__ == "__main__":
    main()
