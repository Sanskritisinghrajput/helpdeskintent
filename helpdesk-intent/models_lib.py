"""TF-IDF, Bi-GRU/Bi-LSTM and BERT intent classifiers behind one interface.

Every model exposes:
    fit(...)                 train
    predict_proba(texts)     -> np.ndarray (n, n_classes) of probabilities
    save(dir) / load(dir)    persistence
"""
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from config import MODEL_DIR, SEED


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1. TF-IDF + Logistic Regression
# =============================================================================
class TfidfModel:
    name = "tfidf"

    def __init__(self, labels):
        self.labels = labels
        self.pipe = None

    def fit(self, texts, y, **_):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline, make_union

        features = make_union(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True),
        )
        self.pipe = make_pipeline(features, LogisticRegression(C=5.0, max_iter=3000))
        self.pipe.fit(list(texts), list(y))
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba(list(texts))

    def save(self, d: Path):
        import joblib
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, d / "tfidf.joblib")

    @classmethod
    def load(cls, d: Path, labels):
        import joblib
        m = cls(labels)
        m.pipe = joblib.load(d / "tfidf.joblib")
        return m

    @staticmethod
    def exists(d: Path):
        return (d / "tfidf.joblib").exists()


# =============================================================================
# 2. Bi-GRU / Bi-LSTM (trained from scratch)
# =============================================================================
TOKEN_RE = re.compile(r"[a-z0-9']+")
PAD, UNK = 0, 1


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def _build_net(vocab_size, n_classes, cell, emb_dim=128, hidden=128, dropout=0.35):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
            rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
            self.rnn = rnn_cls(emb_dim, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden * 4, n_classes)   # mean-pool + max-pool

        def forward(self, ids, lengths):
            x = self.drop(self.emb(ids))
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
            mask = (ids != PAD).unsqueeze(-1)
            mean = (out * mask).sum(1) / lengths.unsqueeze(-1).clamp(min=1).to(out.dtype)
            mx = out.masked_fill(~mask, -1e9).max(1).values
            return self.fc(self.drop(torch.cat([mean, mx], dim=1)))

    return Net()


class RNNModel:
    def __init__(self, labels, cell="gru", max_len=48):
        assert cell in ("gru", "lstm")
        self.labels, self.cell, self.max_len = labels, cell, max_len
        self.name = cell
        self.vocab, self.net = None, None
        self.device = None

    # ---- encoding ---------------------------------------------------------
    def _encode(self, texts):
        import torch
        seqs = []
        for t in texts:
            ids = [self.vocab.get(w, UNK) for w in tokenize(t)][: self.max_len] or [UNK]
            seqs.append(ids)
        lengths = torch.tensor([len(s) for s in seqs])
        padded = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = torch.tensor(s)
        return padded, lengths

    # ---- training ---------------------------------------------------------
    def fit(self, texts, y, val_texts=None, val_y=None, epochs=25, batch_size=64, lr=2e-3, patience=4, **_):
        import torch
        import torch.nn as nn

        set_seed()
        self.device = get_device()
        freq = {}
        for t in texts:
            for w in tokenize(t):
                freq[w] = freq.get(w, 0) + 1
        self.vocab = {w: i + 2 for i, w in enumerate(sorted(freq))}
        self.net = _build_net(len(self.vocab) + 2, len(self.labels), self.cell).to(self.device)

        y = torch.tensor(list(y))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        best, best_state, bad = -1.0, None, 0
        n = len(texts)
        texts = list(texts)

        for epoch in range(1, epochs + 1):
            self.net.train()
            order = np.random.permutation(n)
            total = 0.0
            for s in range(0, n, batch_size):
                idx = order[s: s + batch_size]
                ids, lengths = self._encode([texts[i] for i in idx])
                drop = (torch.rand(ids.shape) < 0.08) & (ids != PAD)      # word dropout -> robustness
                ids = ids.masked_fill(drop, UNK)
                logits = self.net(ids.to(self.device), lengths.to(self.device))
                loss = loss_fn(logits, y[idx].to(self.device))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            msg = f"  [{self.name}] epoch {epoch:02d} loss {total / n:.4f}"
            if val_texts is not None:
                acc = float((self.predict_proba(val_texts).argmax(1) == np.array(val_y)).mean())
                msg += f"  val_acc {acc:.4f}"
                if acc > best:
                    best, bad = acc, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
            print(msg, flush=True)
            if val_texts is not None and bad >= patience:
                print("  early stopping")
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # ---- inference --------------------------------------------------------
    def predict_proba(self, texts):
        import torch
        self.net.eval()
        texts, out = list(texts), []
        with torch.no_grad():
            for s in range(0, len(texts), 256):
                ids, lengths = self._encode(texts[s: s + 256])
                logits = self.net(ids.to(self.device), lengths.to(self.device))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    # ---- persistence ------------------------------------------------------
    def save(self, d: Path):
        import torch
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "cell": self.cell, "max_len": self.max_len,
                    "vocab_size": len(self.vocab) + 2}, d / f"{self.cell}.pt")
        (d / f"{self.cell}_vocab.json").write_text(json.dumps(self.vocab))

    @classmethod
    def load(cls, d: Path, labels, cell):
        import torch
        ckpt = torch.load(d / f"{cell}.pt", map_location="cpu")
        m = cls(labels, cell=cell, max_len=ckpt["max_len"])
        m.vocab = json.loads((d / f"{cell}_vocab.json").read_text())
        m.device = get_device()
        m.net = _build_net(ckpt["vocab_size"], len(labels), cell).to(m.device)
        m.net.load_state_dict(ckpt["state"])
        m.net.eval()
        return m

    @staticmethod
    def exists(d: Path, cell):
        return (d / f"{cell}.pt").exists() and (d / f"{cell}_vocab.json").exists()


# =============================================================================
# 3. BERT (fine-tuned with Hugging Face transformers)
# =============================================================================
class BertModel:
    name = "bert"

    def __init__(self, labels, max_len=48):
        self.labels, self.max_len = labels, max_len
        self.tok, self.net, self.device = None, None, None

    def fit(self, texts, y, val_texts=None, val_y=None, epochs=4, batch_size=32, lr=3e-5,
            pretrained="bert-base-uncased", **_):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        set_seed()
        self.device = get_device()
        # Tiny checkpoints need a larger learning rate and more epochs than bert-base.
        pretrained_lower = pretrained.lower()
        if "tiny" in pretrained_lower:
            lr = 5e-5 if lr == 3e-5 else lr
            epochs = max(epochs, 8)
        print(f"  [bert] fine-tuning '{pretrained}' on {self.device}, lr={lr:.1e}, epochs={epochs}")
        self.tok = AutoTokenizer.from_pretrained(pretrained)
        self.net = AutoModelForSequenceClassification.from_pretrained(
            pretrained, num_labels=len(self.labels), ignore_mismatched_sizes=True,
            id2label={i: l for i, l in enumerate(self.labels)},
            label2id={l: i for i, l in enumerate(self.labels)},
        ).to(self.device)

        texts, y = list(texts), torch.tensor(list(y))
        n = len(texts)
        steps = epochs * ((n + batch_size - 1) // batch_size)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        best, best_state = -1.0, None

        for epoch in range(1, epochs + 1):
            self.net.train()
            order = np.random.permutation(n)
            total = 0.0
            for s in range(0, n, batch_size):
                idx = order[s: s + batch_size]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.device)
                out = self.net(**enc, labels=y[idx].to(self.device))
                opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
                total += out.loss.item() * len(idx)
            msg = f"  [bert] epoch {epoch}/{epochs} loss {total / n:.4f}"
            if val_texts is not None:
                acc = float((self.predict_proba(val_texts).argmax(1) == np.array(val_y)).mean())
                msg += f"  val_acc {acc:.4f}"
                if acc > best:
                    best = acc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            print(msg, flush=True)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, texts):
        import torch
        self.net.eval()
        texts, out = list(texts), []
        with torch.no_grad():
            for s in range(0, len(texts), 64):
                enc = self.tok(texts[s: s + 64], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.device)
                out.append(torch.softmax(self.net(**enc).logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def save(self, d: Path):
        d = d / "bert"
        d.mkdir(parents=True, exist_ok=True)
        self.net.save_pretrained(d)
        self.tok.save_pretrained(d)

    @classmethod
    def load(cls, d: Path, labels):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        m = cls(labels)
        m.device = get_device()
        m.tok = AutoTokenizer.from_pretrained(d / "bert")
        m.net = AutoModelForSequenceClassification.from_pretrained(d / "bert").to(m.device)
        m.net.eval()
        return m

    @staticmethod
    def exists(d: Path):
        return (d / "bert" / "config.json").exists()


# =============================================================================
# Registry helpers
# =============================================================================
def model_exists(name, d: Path = MODEL_DIR):
    if name == "tfidf":
        return TfidfModel.exists(d)
    if name in ("gru", "lstm"):
        return RNNModel.exists(d, name)
    if name == "bert":
        return BertModel.exists(d)
    return False


def load_model(name, labels, d: Path = MODEL_DIR):
    if name == "tfidf":
        return TfidfModel.load(d, labels)
    if name in ("gru", "lstm"):
        return RNNModel.load(d, labels, name)
    if name == "bert":
        return BertModel.load(d, labels)
    raise ValueError(name)


def timed(fn, *a, **k):
    t = time.time()
    r = fn(*a, **k)
    return r, time.time() - t
