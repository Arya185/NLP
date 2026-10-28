# Run Guide: English + Hinglish Hate Speech Detection

Commands were verified on macOS with Python 3.12.2 (checked 2026-10-28), using scratch copies of exactly the
files in the source repository. Anything not verified is marked **(unverified)**. Run every command from the
project root, the folder that contains `app.py`.

## 1. Overview

The app classifies text into three classes, in this fixed order:

| id | label |
|---|---|
| 0 | HATE |
| 1 | OFFENSIVE |
| 2 | NORMAL |

| Model | Files | Status in a fresh clone |
|---|---|---|
| TF-IDF + calibrated Linear SVM | `models/svm_pipeline.joblib` | **The default model, but the file is not in the repository.** It is excluded pending review (its vocabulary was learned from tweet text, and its redistribution status has not been established). You generate it locally (section 3). |
| MuRIL fine-tuned (`google/muril-base-cased`) | `models/muril_final_model/` | Optional. **Unavailable until its required artifacts and dependencies are installed** (section 5). |

**Consequence:** in a fresh clone the app opens with the SVM selected and marked "unavailable", and it cannot
predict anything until you either generate the SVM (section 3) or install MuRIL and switch to it (section 5).
The default selection is not changed automatically.

### Where things live

| Category | Contents |
|---|---|
| **In the source repository** | `app.py`, `src/`, `train_baseline.py`, `train_muril.py`, `colab_muril_finetune.ipynb`, two earlier exploratory Colab notebooks, `tests/`, `requirements.txt`, `requirements-muril.txt`, `README.md`, `RUN_GUIDE.md`, `THIRD_PARTY_NOTICES.md`, `.gitignore`, `.streamlit/config.toml`. The small MuRIL metadata files (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `label_config.json`, `eval_results.json`) may also be present. They alone do **not** make MuRIL available, because the weights are required. |
| **You must obtain or generate locally** | The SVM model `models/svm_pipeline.joblib` (generate with `train_baseline.py`), and the raw English dataset `data/raw/davidson_raw.csv` (you must obtain it yourself, see section 3). |
| **Separate GitHub Release asset** | `muril-hate-speech-model-v1.zip`: the MuRIL weights and metadata (section 5). |
| **Never committed (git-ignored)** | Raw datasets (`data/raw/`), the MuRIL weight files, the Colab upload bundle `project_src.zip` (created on demand, section 7), local virtual environments, caches, secrets files and local tool state. |

Main source files:

| Path | Purpose |
|---|---|
| `app.py` | Streamlit app (entry point) |
| `src/preprocess.py` | `clean_text` and label maps (shared by training and the app) |
| `src/data.py` | Dataset loading, de-duplication, leakage-free splits |
| `src/muril_model.py` | MuRIL artifact validation, loading, inference |
| `src/explain.py` | Word-level explanation (leave-one-word-out) |
| `train_baseline.py` | Trains the SVM (**overwrites `models/svm_pipeline.joblib`**) |
| `train_muril.py` | MuRIL fine-tuning; requires a GPU (exits on CPU) |
| `colab_muril_finetune.ipynb` | Colab workflow that runs `train_muril.py` and packages the model |

## 2. Python environment

- Use **Python 3.12** (3.12.2 verified). Other versions are **unverified**.
- `requirements.txt` (app, SVM training, tests) and `requirements-muril.txt` (MuRIL) do not pin versions.
- Versions that worked together when this guide was written (a record, not a lock file): streamlit 1.66.1,
  scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, joblib 1.6.0, emoji 2.16.0, pytest 9.2.0, and, for MuRIL,
  torch 2.15.0 and transformers 5.21.0. Installing newer versions from scratch is **unverified**.

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

`requirements-muril.txt` (torch + transformers, a large download) is needed only for MuRIL (section 5).

## 3. Generate the SVM model (prerequisites and steps)

`train_baseline.py` trains the SVM (word and character TF-IDF with a calibrated Linear SVM). Checked against
the code, it needs:

1. **The English dataset file at `data/raw/davidson_raw.csv`**, a CSV with the columns `tweet` and `class`
   (`class`: 0 hate, 1 offensive, 2 neither), relative to the directory you run the script from, so run it from the
   project root.
2. The packages from `requirements.txt` (scikit-learn, pandas, emoji, joblib).
3. A writable `models/` folder. The script creates it if needed.

**You must obtain the dataset yourself, under terms that apply to you.** It is not included in this repository.
The English data comes from Davidson et al. (see `THIRD_PARTY_NOTICES.md`). The tweet text is third-party content
whose redistribution terms were not cleared by this project, so verify that you may access and use it before you
download or use it, and **do not redistribute the raw data**. If the file is missing, `src/data.py` tries to
download it from the dataset authors' GitHub repository (the URL is in the source). That automatic download was
**not exercised during verification (unverified)**. Placing your own lawfully obtained copy at the path above
was verified.

```bash
python train_baseline.py
```

This writes `models/svm_pipeline.joblib` and prints validation and test reports. **It overwrites any existing file
at that path.** It does not store the metrics anywhere.

Verified reproducibility check: running this script in the same environment, with the same dataset file,
produced a model whose predicted probabilities on the 300-row test split were identical to the author's earlier
artifact (maximum difference 0.0). A different dataset copy or different package versions may give different
results (unverified).

## 4. Run the app and the tests

```bash
source .venv312/bin/activate
streamlit run app.py
python -m pytest -q tests
```

Streamlit served the app at `http://localhost:8501`. Use `--server.port <n>` if that port is busy.

- The text box starts **empty**. Nothing is predicted until you click **ANALYSE TEXT**.
- **TF-IDF + Linear SVM** is selected by default, and the app never changes your selection.
- A model that cannot run is labelled "· unavailable". Selecting it shows the reason and setup steps, and no
  other model is substituted.
- Results show the predicted class, three probabilities (HATE, OFFENSIVE, NORMAL), the cleaned text, and
  word-level contributions. Changing the text or the model hides the previous result until you analyse again.

There are 28 tests. Verified results in scratch copies of the repository files:

| Situation | Result |
|---|---|
| Fresh clone: no SVM file, no MuRIL weights | **26 passed, 2 failed.** The failures are `test_svm_real_prediction_three_class_order_and_empty_input` and `test_switching_models_never_shows_stale_result`, which need the real SVM. The app itself starts without errors and the default SVM shows "unavailable". |
| After generating the SVM (section 3) | 28 passed |
| After also installing the MuRIL archive and `requirements-muril.txt` | 28 passed |

Other notes on the tests:

- Tests of the "MuRIL absent" behaviour simulate missing artifacts, so they do not depend on what is in `models/`.
- **No test loads the real MuRIL weights.** The "artifacts present" case is covered by validation of synthetic
  folder structures and a stub-model wiring test. Real loading and inference were checked separately by hand.
- The header comment of `tests/test_app.py` is outdated (it says no trained artifacts exist locally).

## 5. Optional: add the MuRIL model

MuRIL is **unavailable until both** of these are in place:

1. The model artifacts in **`models/muril_final_model/`** (exactly this path, relative to the project root).
2. The packages in `requirements-muril.txt`.

If either is missing, the app marks MuRIL "unavailable" and says why. The weights are about 950 MB, so they are
distributed as a **separate GitHub Release asset**, not in the repository.

### 5.1 Download

Download `muril-hate-speech-model-v1.zip` (883,443,657 bytes) from the release page. If the release also
provides them, download `muril-hate-speech-model-v1.zip.sha256` and `SHA256SUMS.txt` too.

```
<RELEASE_ASSET_URL>      <- PLACEHOLDER. No release has been published yet, so this URL does not exist.
                            Replace it with the real asset URL once it is available.
```

A command-line download such as `curl -L -O "<RELEASE_ASSET_URL>"` is **unverified** (the placeholder is not a
real URL).

### 5.2 Verify the archive checksum

Expected SHA-256 of `muril-hate-speech-model-v1.zip`:

```
e37de017550443908e19bee6df2f06f4086bf3305f3c155d4769091cddc2e04d
```

```bash
shasum -a 256 muril-hate-speech-model-v1.zip
# or, with the downloaded .sha256 file next to the ZIP:
shasum -a 256 -c muril-hate-speech-model-v1.zip.sha256
```

Both commands were verified on macOS. The values must match exactly. If they do not, delete the download and fetch
it again. A rebuilt archive would have a different hash, so also compare with the release notes. On Linux,
`sha256sum` is the usual equivalent **(unverified)**.

### 5.3 Extract into `models/`

The archive holds one top-level folder, `muril_final_model/`. Extract it into `models/`:

```bash
unzip -o muril-hate-speech-model-v1.zip -d models/
```

Verified on macOS: this produced `models/muril_final_model/` with the six files below, and no nested
`muril_final_model/muril_final_model/`. `-o` overwrites existing files, so use it if your clone already contains
the small metadata files (identical to the ones in the archive).

```
models/muril_final_model/
  model.safetensors
  config.json
  tokenizer.json
  tokenizer_config.json
  label_config.json
  eval_results.json
```

### 5.4 Verify the extracted files (optional)

| File | SHA-256 |
|---|---|
| `model.safetensors` | `554a91d66855cb71da9ebb5178b5792d6a5f5c18e85f710c9014c6b903945cf7` |
| `tokenizer.json` | `7c2c3f36618e2bec6c602c1fba5e8f8ebafaba7db5027f94fc39789124afd132` |
| `config.json` | `41bc24f5bb14c17403a060f415b355ff07fd7a00e31e1e07775a515542e1b4b9` |
| `tokenizer_config.json` | `b76c620d0bbc6bc3ef9a15244aab0a04a535fbbad1e52c4ea9b4618aff79a314` |
| `label_config.json` | `b8f292ca406eac82ee721187e454a7b394ba315d0115dc11199367daea8743d1` |
| `eval_results.json` | `b03f15428dfd76bfab001c73007c91f0b1080a10299649e4b9641e30e86fa35e` |

These match the verified release bundle. With `SHA256SUMS.txt` (its paths start with `muril_final_model/`, so run
it from inside `models/`; verified on macOS):

```bash
cd models
shasum -a 256 -c /path/to/SHA256SUMS.txt
cd ..
```

### 5.5 Install the MuRIL dependencies

```bash
source .venv312/bin/activate
pip install -r requirements-muril.txt
```

This installs `torch` and `transformers` (a large download). Restart Streamlit afterwards.

### 5.6 Confirm MuRIL is available

```bash
python -c "import sys; sys.path.insert(0,'src'); import muril_model; print(muril_model.check_artifacts('models/muril_final_model')[:2])"
```

Expected when the files are in place: `(True, 'ok')`. Otherwise it prints `False` and the reason. This checks the
files only; the app additionally needs `torch` and `transformers` to be importable. In the running app, the page
header then reads "2 OF 2 MODELS READY" (when the SVM exists too) and the MuRIL button has no "unavailable" tag.
The default selection stays SVM either way.

`muril_model.check_artifacts` requires these non-empty files: `config.json` (labels must match the project
mapping), `model.safetensors` (or `pytorch_model.bin`), `tokenizer_config.json`, `tokenizer.json` (or `vocab.txt`),
`label_config.json` (labels must match) and `eval_results.json`.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` | The virtual environment is not activated, or `pip install -r requirements.txt` was not run in it. |
| The default SVM shows "unavailable" / "models/svm_pipeline.joblib not found" | Expected in a fresh clone. Generate it (section 3), or install MuRIL (section 5) and switch to it. |
| `train_baseline.py` fails while downloading the dataset | The automatic download needs network access and was not verified. Place your own lawfully obtained copy at `data/raw/davidson_raw.csv` (section 3). |
| `train_baseline.py` fails with a missing column | The dataset file must have the columns `tweet` and `class`. |
| MuRIL "unavailable": "Folder not found" | `models/muril_final_model/` does not exist yet. Follow section 5. |
| "Incomplete model folder ... Missing or empty: ..." | A required file is missing or empty (for example `model.safetensors` when only the small metadata files are present). Re-download and re-extract. |
| After extracting, files are at `models/muril_final_model/muril_final_model/...` | The ZIP was extracted into the wrong folder. Extract with `-d models/` (section 5.3). |
| "label_config.json labels ... do not match" | The saved label order differs from HATE=0, OFFENSIVE=1, NORMAL=2. Do not use that model. |
| "MuRIL artifacts found but a dependency is missing" | `pip install -r requirements-muril.txt` in the active environment, then restart Streamlit. Availability of `torch` wheels for your platform is **unverified**. |
| Warning that `src/preprocess.py` differs from the version the model was trained with | `src/preprocess.py` changed after training (hash mismatch). Predictions may degrade. The release model records hash prefix `6332aa3d`, which matched the repository's file when this guide was written. |
| MuRIL analysis shows a Python exception | Loading or inference failed (for example corrupt weights or low memory). The message is shown and nothing is substituted. Re-verify the checksum. |
| Checksum does not match | Re-download. Do not use a file that fails verification. |
| Port already in use | Stop the other Streamlit (`pkill -f "streamlit run app.py"`) or use `--server.port`. |
| First MuRIL analysis is slow | The 950 MB model loads on first use (about 5 s on the author's Mac CPU), then it is cached. |
| `train_muril.py` exits with "No GPU detected" | By design. Use the Colab notebook. `--allow-cpu` exists but is very slow. |
| 2 tests fail in a fresh clone | Expected while `models/svm_pipeline.joblib` is absent (section 4). |

## 7. Retraining MuRIL (optional; not needed to run the app)

Use `colab_muril_finetune.ipynb` on a Colab T4 GPU. The notebook asks you to upload a `project_src.zip` containing
`src/` and `train_muril.py`. That file is not in the repository; create it with the command printed in the notebook:

```bash
zip -r project_src.zip src train_muril.py -x "*/__pycache__/*" "*.pyc"
```

Then run all cells and extract the model the notebook downloads into `models/`. Training also needs the English
dataset (section 3 prerequisites and terms apply) and downloads `google/muril-base-cased` from Hugging Face.
Real training was done this way once. Whether it reproduces bit for bit is **unverified** (GPU training is not
guaranteed deterministic).

## 8. Reproducibility notes

**Data**
- English: Davidson et al. `labeled_data.csv`, which `src/data.py` can download from the `master` branch of the
  authors' GitHub repository. That URL is **not pinned to a commit**, so the source file could change.
- **Raw datasets are not included in the source repository.** The tweet text is third-party content and its
  redistribution terms were not cleared. X's developer policy (read 2026-10-28) restricts redistributing posts
  beyond IDs; this is not legal advice. The authors' repository being MIT-licensed does not settle that.
  Verify your own right to access and use the data.
- Davidson's coding is used as is: class 0 hate, 1 offensive, 2 neither (mapped to HATE, OFFENSIVE, NORMAL).
- Hinglish: **15 hand-written synthetic sentences** (5 per class) inside `src/data.py`. There is no real
  Hinglish dataset, and no HASOC data is used.

**Preprocessing** (`src/preprocess.py`, shared by training and the app): lowercase; remove URLs, `@mentions` and
`RT`; emoji to text; collapse 3+ repeated characters to 2; strip punctuation; map common Hinglish spellings
(`bsdk` becomes `bhosdike`, etc.). URLs are removed before punctuation is stripped.

**Splits** (`src/data.py`): up to 1000 English rows per class (seed 42); text is cleaned; rows with duplicate
cleaned text are dropped, as are rows whose duplicate text had conflicting labels. English is split 80/10/10,
stratified, seed 42. Hinglish is split 3 train / 1 val / 1 test per class. Resulting sizes: train 2377
(2368 English + 9 Hinglish), val 299, test 300. The code asserts there is no overlap between splits on cleaned
text; near-duplicates with different cleaned text are not detected.

**Seeds:** 42 in data sampling, the splits, `LinearSVC` and `train_muril.py --seed`. **Environment variables:**
none are read. **Downloads:** the English dataset (training only, if the file is absent) and, for MuRIL training,
`google/muril-base-cased` from Hugging Face. Running the app needs no network once the artifacts are in place.

**Metrics** (test split of 300 rows; single run)

| Metric | Provenance | Value |
|---|---|---|
| MuRIL combined macro-F1 | recorded in `eval_results.json` (shipped with the release) | 0.8000 |
| MuRIL English macro-F1 (297 rows) | recorded | 0.8015 |
| MuRIL combined confusion matrix | recorded | `[[76,14,9],[20,76,5],[9,3,88]]` (rows: true HATE, OFFENSIVE, NORMAL) |
| The three MuRIL values above | recomputed once by the author by rebuilding the splits and running the saved model | identical to recorded |
| MuRIL best-checkpoint validation macro-F1 | recorded only | 0.8208 |
| MuRIL Hinglish macro-F1 | recorded | 0.5556 on **3 sentences**, meaningless |
| SVM combined / English macro-F1 | measured on the author's machine; no metrics file is stored | 0.7432 / 0.7476 |
| SVM Hinglish macro-F1 | measured on the author's machine | 0.1667 on **3 sentences**, meaningless |

**Limitations**
- **Hinglish evaluation is insufficient to establish reliable performance.** The Hinglish data is 15 synthetic
  sentences (9 for training, 3 for testing). No Hinglish number here supports any claim about Hinglish
  hate-speech detection.
- English results come from one tweet dataset and one 300-row test split, with no confidence intervals. Behaviour
  on other kinds of text is unknown.
- The models predict Davidson-style labels. Offensive versus hate is the main confusion (20 of 100 OFFENSIVE test
  rows were predicted HATE by MuRIL).
- Re-running MuRIL training, as opposed to evaluating the saved model, was not repeated.
- There is no lock file, and a from-scratch install with newer package versions is unverified.

## 9. Licensing, attribution and safety notes

- The project's original code is under the MIT License (see `LICENSE`; Copyright (c) 2026 Arya Patel), which covers
  only original code. See `THIRD_PARTY_NOTICES.md` for upstream attribution and the open licensing questions.
  Nothing in this project claims that redistribution of tweet text, or of artifacts derived
  from it, has been cleared.
- **Loading an untrusted `.joblib` file (a Python pickle) can execute arbitrary code.** Only load model files you
  generated yourself or obtained from a trusted source, and verify checksums of anything you download.
- The tests, the Hinglish seed data and these documents include offensive and hateful example text.
