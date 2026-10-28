# English + Hinglish Hate Speech Detection

A text-classification workbench that labels a piece of text as **HATE**, **OFFENSIVE** or **NORMAL**, shows
the class probabilities, and highlights the words that influenced the result. It ships a Streamlit web app,
two models (a TF-IDF + Linear SVM baseline and a fine-tuned MuRIL transformer), the training code, and a test
suite. English support is based on a public tweet dataset. **Hinglish support is experimental and not
validated** (see [Hinglish status](#hinglish-status)).

> **Content warning:** the tests, the Hinglish seed data and the documentation include hateful and offensive
> example text.

**Contents:** [Features](#features) · [How it works](#how-it-works) · [Data and labels](#data-and-labels) ·
[Evaluation](#evaluation) · [Repository layout](#repository-layout) · [Setup](#setup) ·
[MuRIL model](#optional-the-muril-model-separate-release-asset) · [Verified status](#verified-status) ·
[Retraining](#retraining) · [Limitations](#limitations-and-responsible-use) · [Security](#security) ·
[Licence](#licence-and-attribution)

## Features

- **Three-class prediction** in a fixed order: HATE (0), OFFENSIVE (1), NORMAL (2), with a probability for
  each class.
- **Two interchangeable models:** TF-IDF + Linear SVM (the default) and fine-tuned MuRIL
  (`google/muril-base-cased`). You choose one in the app. The app never silently substitutes one for the other.
- **Explanations:** word-level highlighting based on leave-one-word-out occlusion, available for both models.
- **Transparent preprocessing:** the app shows the cleaned text the model actually received.
- **Careful UI behaviour:** the text box starts empty, nothing is predicted until you click **ANALYSE TEXT**,
  and a result is only shown for the exact text and model it was computed for.
- **Honest availability handling:** a model that cannot run is labelled "unavailable" with the reason and setup
  steps, instead of failing or falling back.
- **Leakage-aware data splits** and a **28-test** suite.

## How it works

```
text ──► clean_text ──► model ──► 3 probabilities ──► predicted class
                          │                                   │
                          └── leave-one-word-out ─────────────┴──► word highlighting
```

**Preprocessing** (`src/preprocess.py`, shared by training and the app): lowercase; remove URLs, `@mentions`
and `RT`; convert emoji to text; collapse 3+ repeated characters to 2; strip punctuation; map common Hinglish
spellings to one form (for example `bsdk` becomes `bhosdike`). URLs are removed before punctuation is stripped.

| Model | Details |
|---|---|
| **TF-IDF + Linear SVM** (default) | Word TF-IDF (1-2 grams, up to 10,000 features) joined with character TF-IDF (3-5 grams, up to 15,000 features), then a calibrated `LinearSVC` (balanced class weights, 3-fold calibration, seed 42). Trained by `train_baseline.py`. |
| **MuRIL fine-tuned** | `google/muril-base-cased` as a 3-class sequence classifier. Training settings recorded in the model's `label_config.json`: up to 4 epochs, batch size 16, learning rate 2e-5, early-stopping patience 2, max length 128, seed 42. The best checkpoint is chosen by validation macro-F1. Class-weighted loss is used only when the training classes are imbalanced (ratio above 1.25); it was off for this run because the classes were balanced. Trained by `train_muril.py` on a GPU (Colab workflow provided). |

**Explanations** (`src/explain.py`): each word of the cleaned text is removed in turn, and the drop in the
predicted class's probability is the word's contribution. This is model-agnostic. It is an approximation of what
influenced the prediction, not a causal explanation.

## Data and labels

- **English:** Davidson et al., *Automated Hate Speech Detection and the Problem of Offensive Language* (tweets
  labelled hate speech / offensive language / neither). Used as is: 0 hate, 1 offensive, 2 neither, mapped to
  HATE, OFFENSIVE, NORMAL. Up to 1000 rows per class are sampled (seed 42).
- **Hinglish:** only **15 hand-written synthetic sentences** (5 per class) inside `src/data.py`.
- **Cleaning and splits:** text is cleaned, rows whose cleaned text is duplicated are dropped, and rows whose
  duplicates had conflicting labels are dropped. English is split 80/10/10 (stratified, seed 42); Hinglish is
  split 3 train / 1 validation / 1 test per class. The code asserts that no cleaned text appears in more than
  one split.

| Split | Rows | Notes |
|---|---|---|
| Train | 2377 | 2368 English + 9 Hinglish |
| Validation | 299 | |
| Test | 300 | 297 English + 3 Hinglish |

**The raw dataset is not included in this repository.** You must obtain it yourself and independently verify
that you may access and use it (see [Setup](#2-generate-the-svm-model-needed-for-the-default-model)).

## Evaluation

All numbers are for the 300-row test split, from a single run, with no confidence intervals.

| Result | Provenance | Value |
|---|---|---|
| MuRIL, combined macro-F1 | recorded in `eval_results.json` (ships with the Release asset); recomputed once by rebuilding the splits and running the saved model, with identical results | 0.8000 |
| MuRIL, English macro-F1 (297 rows) | same | 0.8015 |
| MuRIL English per class (precision / recall / F1) | recorded | HATE 0.721 / 0.765 / 0.743; OFFENSIVE 0.817 / 0.760 / 0.788; NORMAL 0.870 / 0.879 / 0.874 |
| MuRIL combined confusion matrix (rows: true HATE, OFFENSIVE, NORMAL) | recorded and recomputed | `[[76,14,9],[20,76,5],[9,3,88]]` |
| SVM, combined / English macro-F1 | measured on the author's machine; regenerating the SVM in the same environment gave identical predictions on the test split | 0.7432 / 0.7476 |

The main confusion is OFFENSIVE predicted as HATE (20 of 100 OFFENSIVE test rows for MuRIL). These numbers
describe one tweet dataset and say nothing about other kinds of text. More detail is in
[RUN_GUIDE.md](RUN_GUIDE.md), section 8.

## Hinglish status

**Hinglish performance is experimental and not validated.** The Hinglish training data is a tiny synthetic
seed (9 training sentences) and the Hinglish test set has only 3 examples, so no Hinglish result here supports
any claim about Hinglish hate-speech detection.

## Repository layout

```
.
├── app.py                          Streamlit app (entry point)
├── src/
│   ├── preprocess.py               clean_text and label maps
│   ├── data.py                     dataset loading, de-duplication, splits
│   ├── explain.py                  leave-one-word-out explanations
│   └── muril_model.py              MuRIL artifact validation, loading, inference
├── train_baseline.py               trains the SVM (writes models/svm_pipeline.joblib)
├── train_muril.py                  MuRIL fine-tuning (GPU)
├── colab_muril_finetune.ipynb      Colab workflow for MuRIL
├── colab_training_pipeline.ipynb   earlier exploratory notebook
├── Hate_Speech_Detection_Colab.ipynb   earlier exploratory notebook
├── models/muril_final_model/       small MuRIL metadata; weights come from the Release asset
├── tests/                          test_app.py, test_preprocess.py, test_training_workflow.py
├── .streamlit/config.toml          light theme
├── requirements.txt                app, SVM training and tests
├── requirements-muril.txt          MuRIL inference (torch, transformers)
├── README.md  RUN_GUIDE.md  THIRD_PARTY_NOTICES.md  LICENSE  .gitignore
```

| | Included in this source repository | You must obtain or generate locally | Separate GitHub Release asset |
|---|---|---|---|
| Code, tests, docs, dependency lists | Yes | | |
| Small MuRIL metadata (`config.json`, `tokenizer*.json`, `label_config.json`, `eval_results.json`) | May be present | | Also inside the asset |
| **SVM model** `models/svm_pipeline.joblib` | **No.** Intentionally excluded pending review (its vocabulary was learned from tweet text and its redistribution status has not been established) | **Generate it** with `train_baseline.py` | |
| Raw English dataset (`data/raw/`) | **No** | **You must obtain it** | |
| MuRIL weights `model.safetensors` | **No** | | **Yes**: `muril-hate-speech-model-v1.zip` |

**The SVM is the app's default model, and a fresh clone does not contain it.** Until you generate the SVM (or
install MuRIL and switch to it), the app opens with the SVM selected and marked "unavailable", and it cannot make
predictions. This is intentional and was tested (see [Verified status](#verified-status)).

## Setup

All commands run from the project root (the folder containing `app.py`). They were verified on macOS with
Python 3.12.2; other platforms are unverified.

### 1. Python environment

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

### 2. Generate the SVM model (needed for the default model)

Training needs the Davidson et al. English tweet dataset. **The dataset is not included in this repository, and
you must independently verify that you are allowed to access and use it before downloading or using it.** Tweet
text is third-party content, and its redistribution terms have not been cleared by this project. Do not
redistribute the raw data.

`train_baseline.py` reads `data/raw/davidson_raw.csv` (a CSV with the columns `tweet` and `class`, where `class`
is 0 hate, 1 offensive, 2 neither). You have two options:

- Place your own lawfully obtained copy of the file at `data/raw/davidson_raw.csv`. **Verified.**
- Or leave the file absent. The code then tries to download it from the dataset authors' GitHub repository (the
  URL is in `src/data.py`). **The automatic download was not exercised during verification (unverified).**
  `data/raw/` is git-ignored.

Then run:

```bash
python train_baseline.py
```

This writes `models/svm_pipeline.joblib`. **It overwrites any existing file at that path.**

### 3. Launch the app and run the tests

```bash
streamlit run app.py
python -m pytest -q tests
```

Streamlit served the app at `http://localhost:8501`.

**Using the app:** choose a model with the selector (SVM is selected by default), type or paste text, and click
**ANALYSE TEXT**. You get the predicted class, three probability bars, the cleaned text (in a collapsible
section) and word-level highlighting. **CLEAR** resets the input and result.

## Optional: the MuRIL model (separate Release asset)

MuRIL is **unavailable until both** of these are in place: **(1) the model artifacts in
`models/muril_final_model/`, and (2) the packages in `requirements-muril.txt`.** If either is missing, the app
marks it "unavailable", explains why, and never substitutes another model.

The weights file is about 950 MB, above GitHub's per-file limit for repositories, so it is distributed as a
Release asset.

1. Download `muril-hate-speech-model-v1.zip` from `<RELEASE_ASSET_URL>`.
   **This is a placeholder: no release has been published yet, so this URL does not exist yet.**
2. Verify the archive checksum. The expected SHA-256 for the current archive is:

   ```
   e37de017550443908e19bee6df2f06f4086bf3305f3c155d4769091cddc2e04d
   ```

   ```bash
   shasum -a 256 muril-hate-speech-model-v1.zip     # compare with the value above
   ```

   If the release also provides `muril-hate-speech-model-v1.zip.sha256`, you can run
   `shasum -a 256 -c muril-hate-speech-model-v1.zip.sha256` next to the ZIP. Both commands were verified on macOS.
   On Linux, `sha256sum` is the usual equivalent (unverified). A rebuilt archive would have a different hash, so
   also check the release notes.
3. Extract into `models/`. The archive contains one top-level folder, `muril_final_model/`:

   ```bash
   unzip -o muril-hate-speech-model-v1.zip -d models/
   ```

   The result must be **`models/muril_final_model/`** containing `model.safetensors`, `config.json`,
   `tokenizer.json`, `tokenizer_config.json`, `label_config.json` and `eval_results.json`.
4. Install the MuRIL packages (a large download), then restart the app:

   ```bash
   pip install -r requirements-muril.txt
   ```

Per-file checksums, a command to confirm availability, and troubleshooting are in
[RUN_GUIDE.md, section 5](RUN_GUIDE.md#5-optional-add-the-muril-model).

## Verified status

Checked on macOS with Python 3.12.2, using scratch copies of exactly the files this repository contains:

| Situation | Result |
|---|---|
| Fresh clone: no SVM file, no MuRIL weights | App starts, the default SVM shows "unavailable", nothing can be predicted. 26 of 28 tests pass; the 2 that need the real SVM fail (expected). |
| Clean clone after `train_baseline.py` (dataset file supplied locally) | 28 of 28 tests pass; the SVM predicts. |
| Same, plus the MuRIL archive extracted and `requirements-muril.txt` installed | 28 of 28 tests pass; MuRIL runs when selected. The MuRIL archive used was the local copy, not a download from a URL. |

No test loads the real MuRIL weights; real loading and inference were checked separately by hand.

## Retraining

- **SVM:** `python train_baseline.py` (see [Setup](#2-generate-the-svm-model-needed-for-the-default-model)).
- **MuRIL:** open `colab_muril_finetune.ipynb` in Google Colab with a GPU runtime. The notebook asks you to upload
  a `project_src.zip` (the `src/` folder and `train_muril.py`); create it with the command printed in the notebook.
  It trains, evaluates on the held-out test split, prints a warning when a language has too few test examples,
  and packages the result. `train_muril.py` refuses to run without a GPU unless `--allow-cpu` is passed.
  Whether a retrain reproduces the released weights bit for bit is unverified.

## Limitations and responsible use

- **Hinglish is not validated** (see above).
- English results come from one dataset and one 300-row test split. Behaviour on other platforms, topics, dialects
  or time periods is unknown.
- The models predict the labels of a particular dataset. OFFENSIVE versus HATE is the main confusion, and
  borderline or context-dependent text will be misclassified.
- Text classifiers can reflect biases in their training data. Do not use this as the sole basis for moderating
  content or for decisions about people; treat it as a research and demonstration tool.
- A hateful-looking result is a model output with a probability, not a ruling.

## Security

**Loading an untrusted `.joblib` file (a Python pickle) can execute arbitrary code.** Only load model files you
generated yourself or obtained from a source you trust, and verify checksums of anything you download.

## Licence and attribution

- The project's original code is released under the **MIT License**. See [LICENSE](LICENSE)
  (Copyright (c) 2026 Arya Patel). The MIT licence applies only to original project code; it does not grant rights
  to third-party datasets or model artifacts.
- Third-party models and data have their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
  attribution and for the licensing questions that are still open. **Nothing in this repository claims that
  redistribution of tweet text, or of artifacts derived from it, has been cleared.**
- English data: Davidson, T., Warmsley, D., Macy, M., and Weber, I. (2017). Automated Hate Speech Detection and
  the Problem of Offensive Language. *Proceedings of the International AAAI Conference on Web and Social Media*,
  11(1), 512-515. https://doi.org/10.1609/icwsm.v11i1.14955
- Base model: `google/muril-base-cased` (Khanuja et al., 2021, arXiv:2103.10730), listed as Apache-2.0 on its
  public page; see the notices file for the unverified points.

More documentation: **[RUN_GUIDE.md](RUN_GUIDE.md)** (full setup, troubleshooting, reproducibility notes).
