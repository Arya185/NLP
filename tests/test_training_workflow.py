"""Checks for the Colab training workflow that can run without torch or a GPU.
These do NOT train anything and say nothing about MuRIL quality."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import train_muril  # noqa: E402  (imports numpy only; torch is imported inside main())
from preprocess import ID_TO_LABEL  # noqa: E402


def test_default_output_dir_matches_app_path():
    src = open(os.path.join(ROOT, 'train_muril.py')).read()
    app = open(os.path.join(ROOT, 'app.py')).read()
    assert "'models', 'muril_final_model'" in src and "'models', 'muril_final_model'" in app


def test_warmup_steps_is_explicit_and_scales():
    assert train_muril.warmup_steps_for(2377, 16, 4) == int(0.1 * 149 * 4)
    assert train_muril.warmup_steps_for(10, 16, 1) == 0


def test_evaluate_subset_flags_small_samples_and_is_json_serialisable(capsys):
    hi = pd.DataFrame({'label': [0, 1, 2], 'language': 'hinglish'})
    r = train_muril.evaluate_subset('hinglish test', hi, np.array([2, 2, 2]))
    assert r['reliable'] is False and r['n'] == 3
    assert 'WARNING' in capsys.readouterr().out
    en = pd.DataFrame({'label': [0, 1, 2] * 20, 'language': 'english'})
    r2 = train_muril.evaluate_subset('english test', en, np.array([0, 1, 2] * 20))
    assert r2['reliable'] is True and r2['macro_f1'] == 1.0
    assert r2['confusion_matrix'] == [[20, 0, 0], [0, 20, 0], [0, 0, 20]]
    json.dumps({'a': r, 'b': r2}, default=train_muril._jsonable)
    json.dumps(np.array([1, 2]), default=train_muril._jsonable)


def test_class_weight_threshold_constant_and_label_order():
    assert train_muril.WEIGHT_IF_RATIO_ABOVE > 1 and train_muril.TARGET_NAMES == [ID_TO_LABEL[i] for i in range(3)]


def test_notebook_structure_and_honesty():
    nb = json.load(open(os.path.join(ROOT, 'colab_muril_finetune.ipynb')))
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    text = '\n'.join(''.join(c['source']) for c in nb['cells'])
    assert all(not c.get('outputs') for c in nb['cells'] if c['cell_type'] == 'code')   # no fabricated results
    assert 'torch.cuda.is_available()' in code                                           # GPU required
    assert 'train_muril.py' in code and 'models/muril_final_model' in code
    assert 'check_artifacts' in code and 'muril_final_model.zip' in code
    assert '__pycache__' in text                                                         # zip command excludes pycache
    assert 'not run yet' in text.lower() and 'synthetic' in text.lower()
