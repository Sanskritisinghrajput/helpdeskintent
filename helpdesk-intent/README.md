# Helpdesk Intent Classification Assistant

Routes natural-language helpdesk queries to the right desk (**account, certification, complaint, process, technical**) and hands them to a human when the model is unsure.

* **Models compared:** TF-IDF + Logistic Regression, Bi-GRU, Bi-LSTM, fine-tuned BERT
* **Confidence-based fallback:** if the top probability is below a threshold, the query goes to a human agent. The threshold is tuned on validation data that includes genuine out-of-scope queries (weather, jokes, sports) the models never train on.
* **Web UI:** Flask + vanilla JS, navy blue and deep cherry-burgundy theme. Tabs: Classify, Compare models, Benchmarks, How it works.

## 1. Setup (VS Code)

1. Open this folder in VS Code (`File > Open Folder`). The `.vscode` folder gives you a navy/burgundy editor theme and run configurations.
2. Open a terminal (`Ctrl + ~`) and create an environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

If `pip install torch` is slow or huge, use the CPU build from https://pytorch.org/get-started/locally/ first, then `pip install -r requirements.txt`.

## 2. Get the data

**Option A - built-in helpdesk corpus (works offline, default)**

```bash
python prepare_data.py
```

This generates about 3,500 labelled institutional-helpdesk queries plus 360 out-of-scope queries. It is templated, so it is *easy*: expect 95-100% accuracy for most models. Use it to see the pipeline work, then switch to real data for a meaningful comparison.

**Option B - real CLINC150 data from Kaggle**

1. Create a Kaggle account, go to *Settings > API > Create New Token*, and put the downloaded `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).
2. On Kaggle, search "CLINC150" and pick a dataset that contains `data_full.json`. Copy its slug (`owner/dataset-name`).
3. Run:

```bash
python prepare_data.py --kaggle owner/dataset-name --mapping data/clinc_mapping_example.json --augment-synthetic
```

Or download `data_full.json` yourself and use `--source clinc --path data_full.json`. The original data is also at https://github.com/clinc/oos-eval.

CLINC150 has no "certification" intents, so `--augment-synthetic` adds the built-in corpus for full category coverage. Its `oos` queries become the real out-of-scope set for the fallback. Edit `data/clinc_mapping_example.json` to change how CLINC intents map to your categories.

**Option C - any other Kaggle CSV (SNIPS, Banking77, a helpdesk ticket dataset...)**

```bash
python prepare_data.py --source csv --path my.csv --text-col text --label-col intent --mapping my_map.json --augment-synthetic
```

`my_map.json` maps each intent in the CSV to one of your categories, e.g. `{"AddToPlaylist": "technical"}`.

## 3. Train and compare

```bash
python train.py                     # TF-IDF, GRU, LSTM, BERT
python train.py --models tfidf gru lstm        # skip BERT
python train.py --bert-model distilbert-base-uncased --bert-epochs 3   # lighter BERT
python train.py --models bert --bert-model prajjwal1/bert-tiny --bert-epochs 8 --bert-lr 5e-5  # tiny CPU model
```

BERT downloads pretrained weights from Hugging Face on first run (about 440 MB for `bert-base-uncased`). On CPU, `bert-base` takes roughly 10-20 minutes on the built-in corpus; a GPU takes about a minute.

After each model it prints accuracy, macro-F1, latency and the fallback numbers, and saves everything to `models/results.json`.

## 4. Run the website

```bash
python app.py
```

Open http://127.0.0.1:5000. (In VS Code you can also press `F5` and choose **Run web app**.)

Try: *"I was charged twice for the exam fee"*, *"How do I get a duplicate degree certificate?"*, and *"What's the weather today?"* (should fall back to a human).

## How the fallback works

Each model returns class probabilities; the largest is the **confidence**. If `confidence < threshold`, the query is routed to a human agent.
`train.py` sweeps thresholds from 0.05 to 0.99 and picks, **on the validation set**, the one that best balances
(a) in-scope queries answered *correctly* and (b) out-of-scope queries sent to a human. Reported numbers are on the untouched **test** set. You can override the threshold with the slider in the UI.

## Project layout

```
config.py          labels, desk names, defaults
prepare_data.py    built-in corpus / CLINC150 / Kaggle CSV -> data/*.csv
models_lib.py      TF-IDF, GRU/LSTM, BERT classes (shared interface)
train.py           training, evaluation, threshold tuning -> models/results.json
app.py             Flask API + web server
templates/ static/ the navy & burgundy front end
.vscode/           editor theme + run configs
```

## API

`POST /api/predict` `{ "text": "...", "model": "bert", "threshold": 0.7 }`
`POST /api/compare` `{ "text": "..." }`
`GET /api/status`, `GET /api/metrics`

## Honest notes

* Softmax confidence from neural networks tends to be over-confident on out-of-scope text. That is why the threshold is tuned on real out-of-scope examples rather than fixed at 0.5. For stronger rejection you could add temperature scaling or an energy score.
* The built-in corpus uses templates; scores on it overstate real-world accuracy. Compare models on CLINC150 or your own helpdesk logs.
