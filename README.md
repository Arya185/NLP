# English + Hinglish Hate Speech Detection

A small text-classification workbench. It labels a piece of text as **HATE**, **OFFENSIVE** or
**NORMAL**, shows the class probabilities, and highlights the words that influenced the result.
English support is based on a public tweet dataset. **Hinglish support is experimental and not
validated** (see below).

> **Content warning:** the tests, the Hinglish seed data and the documentation include hateful and
> offensive example text.

## Components

| Part | What it is |
|---|---|
| `app.py` | Streamlit web app (the entry point) |
| `src/` | Shared preprocessing and label mapping, data loading and splits, MuRIL loading, word-level explanations |
| `train_baseline.py` | Trains the TF-IDF + calibrated Linear SVM model (the app's **default** model) |
| `train_muril.py`, `colab_muril_finetune.ipynb` | MuRIL fine-tuning (needs a GPU; Colab workflow provided) |
| `tests/` | Test suite |
| `RUN_GUIDE.md` | Full setup, MuRIL download and verification, troubleshooting, reproducibility notes |
| `THIRD_PARTY_NOTICES.md` | Upstream model and dataset attribution, and open licensing questions |

Built with Python 3.12, Streamlit, scikit-learn and, for the optional MuRIL model, PyTorch and
Hugging Face Transformers.

## What you get, and what you must obtain or generate

| | Included in this source repository | You must obtain or generate locally | Separate GitHub Release asset |
|---|---|---|---|
| Code, tests, docs, dependency lists | Yes | | |
| Small MuRIL metadata (`config.json`, `tokenizer*.json`, `label_config.json`, `eval_results.json`) | May be present | | Also inside the asset |
| **SVM model** `models/svm_pipeline.joblib` | **No.** Intentionally excluded pending review (its vocabulary was learned from tweet text and its redistribution status has not been established) | **Generate it** with `train_baseline.py` | |
| Raw English dataset | **No** | **You must obtain it** (see below) | |
| MuRIL weights `model.safetensors` | **No** | | **Yes**: `muril-hate-speech-model-v1.zip` |

**The SVM is the app's default model, and a fresh clone does not contain it.** Until you generate the
SVM (or install MuRIL and switch to it), the app opens with the SVM selected and marked
"unavailable", and it cannot make predictions. This is intentional and was tested (see "Verified status").

## Setup

All commands run from the project root (the folder containing `app.py`). They were verified on macOS
with Python 3.12.2; other platforms are unverified.

### 1. Python environment

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

### 2. Generate the SVM model (needed for the default model)

Training needs the Davidson et al. English tweet dataset. **The dataset is not included in this repository,
and you must independently verify that you are allowed to access and use it before downloading or
using it.** Tweet text is third-party content, and its redistribution terms have not been cleared by this
project. Do not redistribute the raw data.

`train_baseline.py` reads `data/raw/davidson_raw.csv` (a CSV with the columns `tweet` and `class`, where
`class` is 0 hate, 1 offensive, 2 neither). You have two options:

- Place your own lawfully obtained copy of the file at `data/raw/davidson_raw.csv`. **Verified.**
- Or leave the file absent. The code then tries to download it from the dataset authors' GitHub repository
  (the URL is in `src/data.py`). **The automatic download was not exercised during verification
  (unverified).** `data/raw/` is git-ignored.

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

Streamlit served the app at `http://localhost:8501`. The text box starts empty, and nothing is predicted
until you click **ANALYSE TEXT**.

## Optional: the MuRIL model (separate Release asset)

MuRIL (`google/muril-base-cased`, fine-tuned) is **unavailable until both** of these are in place:
**(1) the model artifacts in `models/muril_final_model/`, and (2) the packages in
`requirements-muril.txt`.** If either is missing, the app marks it "unavailable", explains why, and never
substitutes another model.

The weights file is about 950 MB, above GitHub's per-file limit for repositories, so it is distributed
as a Release asset.

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
   `shasum -a 256 -c muril-hate-speech-model-v1.zip.sha256` next to the ZIP. Both commands were verified on
   macOS. On Linux, `sha256sum` is the usual equivalent (unverified). A rebuilt archive would have a different
   hash, so also check the release notes.
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

## Hinglish status

**Hinglish performance is experimental and not validated.** The Hinglish training data is a tiny
synthetic seed of 15 hand-written sentences and the Hinglish test set has only 3 examples, so no
Hinglish result here supports any claim about Hinglish hate-speech detection. Evaluation details, and
which numbers were recorded versus independently reproduced, are in section 8 of
[RUN_GUIDE.md](RUN_GUIDE.md).

## Security

**Loading an untrusted `.joblib` file (a Python pickle) can execute arbitrary code.** Only load model files
you generated yourself or obtained from a source you trust, and verify checksums of anything you download.

## Licence and attribution

- **No `LICENSE` file has been added yet.** Until one is added, no licence is granted for this project's
  code.
- Third-party models and data have their own terms. See
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and for the licensing questions that are
  still open. **Nothing in this repository claims that redistribution of tweet text, or of artifacts derived
  from it, has been cleared.**
# NLP
