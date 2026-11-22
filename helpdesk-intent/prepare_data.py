"""Build the helpdesk intent dataset.

Sources
-------
synthetic : (default) built-in institutional helpdesk corpus, works offline.
clinc     : the real CLINC150 data (data_full.json). Get it from Kaggle
            (any CLINC150 mirror), the official repo github.com/clinc/oos-eval,
            or HuggingFace. Its intents are mapped onto helpdesk categories with
            a JSON mapping file (see data/clinc_mapping_example.json).
csv       : any CSV with a text column and an intent column (e.g. a Kaggle
            SNIPS / banking77 / helpdesk file) + a JSON mapping file.

Examples
--------
python prepare_data.py
python prepare_data.py --source clinc --path data_full.json \
       --mapping data/clinc_mapping_example.json --augment-synthetic
python prepare_data.py --kaggle <owner>/<dataset-slug> --mapping my_map.json --augment-synthetic
python prepare_data.py --source csv --path my.csv --text-col text --label-col intent --mapping my_map.json

Output (in ./data): train.csv val.csv test.csv val_oos.csv test_oos.csv labels.json
The *_oos files are out-of-scope queries used only to tune/evaluate the
confidence-based fallback. The models never train on them.
"""
import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, DEFAULT_LABELS, SEED

OOS_NAMES = {"oos", "out_of_scope", "out-of-scope", "other", "none"}

# --------------------------------------------------------------------------
# Synthetic institutional helpdesk corpus
# --------------------------------------------------------------------------
PORTALS = ["student portal", "exam portal", "LMS", "campus portal", "admission portal", "library portal"]
DOCS = ["degree certificate", "course completion certificate", "bonafide certificate", "transcript",
        "migration certificate", "provisional certificate", "marksheet", "experience letter",
        "character certificate", "diploma"]
FEES = ["exam fee", "tuition fee", "registration fee", "hostel fee", "late fee", "application fee"]
FORMS = ["admission form", "exam form", "scholarship form", "hostel form", "re-evaluation form"]

TEMPLATES = {
    "account": [
        "i forgot my password for the {portal}",
        "how do i reset my {portal} password",
        "unable to log in to my {portal} account",
        "my password is not working anymore",
        "please send me a password reset link",
        "my account is locked after too many login attempts",
        "my {portal} account got blocked",
        "my login has been disabled since yesterday",
        "i cannot access my account",
        "how can i change my registered email address",
        "update my phone number on my profile",
        "i want to correct my name in my account profile",
        "i want to change my mobile number linked to the {portal}",
        "i am not receiving the otp on my phone",
        "verification code is not coming to my email",
        "the otp expired before i could enter it",
        "i want to delete my {portal} account",
        "please deactivate my student account",
        "i forgot my username",
        "what is my user id for the {portal}",
        "help me recover my login id",
        "two factor authentication is asking for a code i do not have",
        "how do i enable two step verification on my account",
        "my profile photo is not updating",
    ],
    "certification": [
        "how do i apply for a {doc}",
        "i need a {doc} urgently",
        "where can i request my {doc}",
        "i would like to request a {doc} for my visa",
        "what is the status of my {doc} request",
        "when will my {doc} be issued",
        "has my {doc} been dispatched yet",
        "how can an employer verify my {doc}",
        "i need third party verification of my {doc}",
        "my name is spelled wrong on my {doc}",
        "please correct the date of birth on my {doc}",
        "the grade printed on my {doc} is incorrect",
        "i lost my original {doc} and need a duplicate",
        "how do i get a duplicate {doc}",
        "please reissue my {doc}",
        "i want my {doc} attested",
        "how to get an apostille for my {doc}",
        "i need a notarized copy of my {doc}",
        "what is the fee for a {doc}",
        "is a digital copy of my {doc} available",
        "how many copies of the {doc} can i request",
    ],
    "complaint": [
        "i applied three weeks ago and still have no response",
        "my request has been pending for a very long time",
        "this is taking far too long to resolve",
        "the staff at the helpdesk was rude to me",
        "nobody answers my calls at the office",
        "the support agent was completely unhelpful",
        "i was charged twice for the {fee}",
        "the wrong amount was deducted for my {fee}",
        "i want a refund for the {fee} i paid",
        "i am very unhappy with the service i received",
        "this is unacceptable and i want to file a formal complaint",
        "please escalate my issue to a senior manager",
        "i have complained three times and nothing has happened",
        "i complained about my {doc} not arriving and no one cares",
        "my account problem was never resolved and i am frustrated",
        "the portal keeps failing and nobody takes responsibility",
        "i am disappointed with how my {form} was handled",
        "no one has replied to my emails for two weeks",
        "i want to report misbehaviour by an office employee",
        "the fee was deducted but never reflected, this is a scam",
    ],
    "process": [
        "what is the procedure to apply for admission",
        "what are the steps to register for exams",
        "how do i enroll in a new course",
        "which documents are needed for registration",
        "what is the last date to pay the {fee}",
        "when does registration close this semester",
        "how much is the {fee}",
        "what are the office timings",
        "am i eligible to apply for the scholarship",
        "what are the eligibility criteria for this programme",
        "what are the requirements to transfer to another branch",
        "how long does the approval process take",
        "what is the turnaround time for document verification",
        "how many days does it take to process the {form}",
        "i would like to book an appointment with the registrar",
        "how can i schedule a visit to the office",
        "when can i meet the counselor",
        "how do i withdraw from a course",
        "what is the process to apply for a leave of absence",
        "how do i change my elective subject",
        "where do i submit the {form}",
        "is there a waiting list for hostel rooms",
    ],
    "technical": [
        "the {portal} is showing error 500",
        "the page is not loading at all",
        "the website keeps crashing when i open it",
        "i see a blank page after logging in",
        "i can't upload my documents",
        "file upload fails with an error message",
        "the pdf upload size limit error keeps appearing",
        "my payment failed but the money was deducted",
        "the payment page is stuck on loading",
        "the transaction timed out while paying the {fee}",
        "the mobile app keeps closing by itself",
        "the app is not installing on my phone",
        "the app shows a network error",
        "i am unable to download my admit card",
        "the download link for the pdf is broken",
        "my hall ticket is not opening",
        "does the {portal} work on safari",
        "the captcha is not showing up",
        "the site is extremely slow on my phone",
        "the video class does not load",
        "i get a certificate error when opening the {portal}",
        "the submit button does nothing on the {form}",
    ],
}

OOS_TEMPLATES = [
    "what is the weather like today", "tell me a joke", "who won the cricket match yesterday",
    "give me a recipe for biryani", "what are the best movies of this year", "translate hello into french",
    "play some relaxing music", "what is the price of gold today", "how tall is mount everest",
    "who is the prime minister", "book me a flight to delhi", "suggest a good gym workout",
    "what is the meaning of life", "how do i tie a tie", "what is the bitcoin price",
    "recommend a good restaurant nearby", "how many calories are in a banana", "what time is it in tokyo",
    "who painted the mona lisa", "can you sing a song", "what is the capital of australia",
    "how do i lose weight fast", "what is the latest iphone", "tell me a bedtime story",
    "how far is the moon", "who is your favourite footballer", "set an alarm for six am",
    "what should i cook for dinner", "how do i start a vegetable garden", "explain black holes to me",
    "what are some tips for a job interview at a bank", "is it going to rain this weekend",
    "how do i fix a leaking tap", "what is the stock market doing", "recommend a good book to read",
    "who invented the telephone", "what is 25 times 17", "tell me about the history of rome",
    "how do i make coffee at home", "what is the speed of light",
]

PREFIXES = ["", "", "", "hi ", "hello, ", "hey, ", "please ", "excuse me, ", "good morning, ",
            "sir, ", "madam, ", "quick question: ", "i need help. ", "hi team, "]
SUFFIXES = ["", "", "", "", " please", " thanks", " thank you", " asap", " urgently",
            " can you help", " pls help", " kindly guide me", " ?"]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 1)
    mode = rng.choice(["swap", "drop", "dup"])
    if mode == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if mode == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def _noisy(core: str, rng: random.Random) -> str:
    text = rng.choice(PREFIXES) + core + rng.choice(SUFFIXES)
    words = text.split()
    for i, w in enumerate(words):
        if rng.random() < 0.04:
            words[i] = _typo(w, rng)
    if rng.random() < 0.05 and len(words) > 4:      # drop a random word
        words.pop(rng.randrange(len(words)))
    text = " ".join(words).replace(" ?", "?").strip()
    if rng.random() < 0.3:
        text = text.capitalize()
    return text


def _expand(template: str, rng: random.Random):
    def fill(m):
        slot = m.group(1)
        pool = {"portal": PORTALS, "doc": DOCS, "fee": FEES, "form": FORMS}[slot]
        return rng.choice(pool)
    return re.sub(r"\{(\w+)\}", fill, template)


def build_synthetic(per_class: int = 700, oos_total: int = 360, seed: int = SEED):
    rng = random.Random(seed)
    rows = []
    for label, templates in TEMPLATES.items():
        seen, tries = set(), 0
        while len(seen) < per_class and tries < per_class * 40:
            tries += 1
            text = _noisy(_expand(rng.choice(templates), rng), rng)
            if text.lower() not in seen:
                seen.add(text.lower())
                rows.append((text, label))
    oos, seen, tries = [], set(), 0
    while len(seen) < oos_total and tries < oos_total * 40:
        tries += 1
        text = _noisy(rng.choice(OOS_TEMPLATES), rng)
        if text.lower() not in seen:
            seen.add(text.lower())
            oos.append((text, "oos"))
    return pd.DataFrame(rows, columns=["text", "label"]), pd.DataFrame(oos, columns=["text", "label"])


# --------------------------------------------------------------------------
# Real datasets (Kaggle / CLINC150 / any CSV)
# --------------------------------------------------------------------------
def load_mapping(path):
    if not path:
        raise SystemExit("This source needs --mapping <json> mapping dataset intents -> helpdesk categories.")
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k.lower(): v for k, v in m.items()}


def load_clinc(path, mapping):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    def to_df(key):
        rows = []
        for text, intent in d.get(key, []):
            intent = intent.lower()
            if intent in OOS_NAMES:
                rows.append((text, "oos"))
            elif intent in mapping:
                rows.append((text, mapping[intent]))
        return pd.DataFrame(rows, columns=["text", "label"])
    train = pd.concat([to_df("train"), to_df("oos_train").assign(label="oos")])
    val = pd.concat([to_df("val"), to_df("oos_val").assign(label="oos")])
    test = pd.concat([to_df("test"), to_df("oos_test").assign(label="oos")])
    return train, val, test


def load_csv(path, text_col, label_col, mapping):
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna()
    df["label"] = df["label"].astype(str).str.lower().map(lambda x: "oos" if x in OOS_NAMES else mapping.get(x))
    return df.dropna().reset_index(drop=True)


def kaggle_download(slug):
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("pip install kagglehub  (and set up your Kaggle API token) to use --kaggle")
    folder = Path(kagglehub.dataset_download(slug))
    print("Downloaded to", folder)
    for pattern in ("**/data_full.json", "**/*.json", "**/*.csv"):
        hits = sorted(folder.glob(pattern))
        if hits:
            print("Using", hits[0])
            return hits[0]
    raise SystemExit(f"No json/csv found in {folder}")


# --------------------------------------------------------------------------
def split_in_scope(df, seed=SEED):
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return train, val, test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "clinc", "csv"], default="synthetic")
    ap.add_argument("--path", help="local data_full.json or csv")
    ap.add_argument("--kaggle", help="Kaggle dataset slug, e.g. owner/dataset-name (needs kagglehub)")
    ap.add_argument("--mapping", help="JSON {dataset_intent: helpdesk_category}")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="intent")
    ap.add_argument("--augment-synthetic", action="store_true",
                    help="add the built-in helpdesk corpus so every category (e.g. certification) is covered")
    ap.add_argument("--per-class", type=int, default=700)
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    syn_in, syn_oos = build_synthetic(per_class=args.per_class)

    if args.kaggle:
        args.path = str(kaggle_download(args.kaggle))
        args.source = "clinc" if args.path.endswith(".json") else "csv"

    if args.source == "synthetic":
        train, val, test = split_in_scope(syn_in)
        oos_all = syn_oos
        oos_val, oos_test = train_test_split(oos_all, test_size=0.5, random_state=SEED)
    else:
        mapping = load_mapping(args.mapping)
        if args.source == "clinc":
            tr, va, te = load_clinc(args.path, mapping)
        else:
            all_df = load_csv(args.path, args.text_col, args.label_col, mapping)
            in_scope = all_df[all_df.label != "oos"]
            tr, va, te = split_in_scope(in_scope)
            oos_df = all_df[all_df.label == "oos"]
            if len(oos_df) > 10:
                ov, ot = train_test_split(oos_df, test_size=0.5, random_state=SEED)
                va, te = pd.concat([va, ov]), pd.concat([te, ot])
        train, val, test = (x[x.label != "oos"] for x in (tr, va, te))
        oos_val, oos_test = va[va.label == "oos"], te[te.label == "oos"]
        if args.augment_synthetic or len(oos_val) < 20:
            s_tr, s_va, s_te = split_in_scope(syn_in)
            train, val, test = pd.concat([train, s_tr]), pd.concat([val, s_va]), pd.concat([test, s_te])
            s_ov, s_ot = train_test_split(syn_oos, test_size=0.5, random_state=SEED)
            oos_val, oos_test = pd.concat([oos_val, s_ov]), pd.concat([oos_test, s_ot])

    labels = sorted(train["label"].unique())
    for name, df in [("train", train), ("val", val), ("test", test), ("val_oos", oos_val), ("test_oos", oos_test)]:
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"{name:9s} {len(df):6d} rows")
    (DATA_DIR / "labels.json").write_text(json.dumps(labels, indent=2))
    print("labels:", labels)
    print("Counts per class (train):")
    print(train["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
