"""Optional MuRIL inference. Imports torch/transformers lazily so the app works without them.

Expected artifact folder (written by train_muril.py, e.g. models/muril_final_model/):
    config.json, model.safetensors | pytorch_model.bin, tokenizer_config.json,
    vocab.txt | tokenizer.json, label_config.json, eval_results.json
"""
import hashlib
import json
import os

from preprocess import ID_TO_LABEL, LABEL_TO_ID

WEIGHT_FILES = ('model.safetensors', 'pytorch_model.bin')
TOKENIZER_VOCAB_FILES = ('vocab.txt', 'tokenizer.json')
DEFAULT_MAX_LEN = 128


def _nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def check_artifacts(model_dir):
    """Validate the artifact folder. Returns (ok, message, label_config). Never raises."""
    if not os.path.isdir(model_dir):
        return False, f'Folder not found: {model_dir}', None
    j = lambda n: os.path.join(model_dir, n)  # noqa: E731
    missing = []
    if not _nonempty(j('config.json')):
        missing.append('config.json')
    if not any(_nonempty(j(w)) for w in WEIGHT_FILES):
        missing.append('model.safetensors (or pytorch_model.bin)')
    if not _nonempty(j('tokenizer_config.json')):
        missing.append('tokenizer_config.json')
    if not any(_nonempty(j(v)) for v in TOKENIZER_VOCAB_FILES):
        missing.append('vocab.txt (or tokenizer.json)')
    for name in ('label_config.json', 'eval_results.json'):
        if not _nonempty(j(name)):
            missing.append(name)
    if missing:
        return False, f'Incomplete model folder {model_dir}. Missing or empty: ' + ', '.join(missing), None

    try:
        cfg = _read_json(j('label_config.json'))
        id2label = {int(k): v for k, v in cfg['id2label'].items()}
        label2id = {k: int(v) for k, v in cfg['label2id'].items()}
        _read_json(j('eval_results.json'))
        hf = _read_json(j('config.json'))
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as e:
        return False, f'Model folder has an unreadable or malformed JSON file ({type(e).__name__}: {e})', None
    if id2label != ID_TO_LABEL or label2id != LABEL_TO_ID:
        return False, (f'label_config.json labels {id2label} do not match the project mapping {ID_TO_LABEL}'), cfg
    hf_id2label = hf.get('id2label')
    if hf_id2label is not None:
        try:
            if {int(k): v for k, v in hf_id2label.items()} != ID_TO_LABEL:
                return False, f'config.json id2label {hf_id2label} does not match the project mapping {ID_TO_LABEL}', cfg
        except (ValueError, AttributeError):
            return False, 'config.json id2label is malformed', cfg
    return True, 'ok', cfg


def artifact_signature(model_dir):
    """Cheap fingerprint (name, size, mtime of key files) so a cached model reloads when artifacts change."""
    sig = []
    for n in ('config.json', 'label_config.json') + WEIGHT_FILES:
        p = os.path.join(model_dir, n)
        if os.path.exists(p):
            st = os.stat(p)
            sig.append((n, st.st_size, int(st.st_mtime)))
    return tuple(sig)


def preprocess_matches(cfg, preprocess_path):
    """True/False if the training-time preprocess hash equals the local file; None if unknown."""
    want = (cfg or {}).get('preprocess_sha256')
    if not want:
        return None
    with open(preprocess_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest() == want


def load(model_dir):
    """Load tokenizer+model. Raises ImportError if torch/transformers are not installed."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    n_out = model.config.num_labels
    if n_out != len(ID_TO_LABEL):
        raise ValueError(f'Model has {n_out} outputs, expected {len(ID_TO_LABEL)} (HATE, OFFENSIVE, NORMAL)')
    return tok, model, torch


def make_proba_fn(tok, model, torch, max_len=DEFAULT_MAX_LEN):
    """Return f(list[str]) -> (n, 3) probabilities in project class order (HATE, OFFENSIVE, NORMAL).

    Column i is class id i: train_muril.py builds the model with id2label=ID_TO_LABEL, and
    check_artifacts verifies the saved mapping equals the project mapping.
    """
    import numpy as np

    def proba(texts):
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(list(texts[i:i + 32]), truncation=True, max_length=max_len, padding=True, return_tensors='pt')
            with torch.no_grad():
                out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(out)
    return proba
