"""App-level tests via Streamlit AppTest. MuRIL with a real trained model is NOT covered here:
no trained artifacts or torch exist locally. The stub test only checks the app's wiring."""
import os
import sys
import types

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import muril_model  # noqa: E402

SVM, MURIL = 'TF-IDF + Linear SVM', 'MuRIL Fine-Tuned'


def run_app():
    return AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=90).run()


def options(at):
    return list(at.radio(key='model_choice').options)


def analyse(at, text):
    at.text_area(key='input_text').set_value(text).run()
    return at.button(key='analyse').click().run()


def text_of(at):
    return ' '.join(m.value for m in at.markdown) + ' ' + ' '.join(e.value for e in at.error)


@pytest.fixture
def no_muril(monkeypatch):
    """Simulate absent MuRIL artifacts so these tests do not depend on what is installed in models/."""
    monkeypatch.setattr(muril_model, 'check_artifacts',
                        lambda d: (False, f'Folder not found: {d}', None))


def test_missing_muril_svm_default_and_marked_unavailable(no_muril):
    at = run_app()
    assert not at.exception
    assert at.radio(key='model_choice').value == SVM          # defaults to SVM
    assert options(at) == [f'{SVM}', f'{MURIL} · unavailable'] or 'unavailable' in options(at)[1]
    assert at.text_area(key='input_text').value == ''          # empty on load
    assert at.session_state['result'] is None                  # no automatic prediction


def test_svm_real_prediction_three_class_order_and_empty_input():
    at = run_app()
    at.button(key='analyse').click().run()                     # empty input
    assert at.session_state['result']['status'] == 'empty' and 'probs' not in at.session_state['result']
    at = analyse(at, 'I love this lovely weather')
    r = at.session_state['result']
    assert r['status'] == 'ok' and len(r['probs']) == 3 and abs(sum(r['probs']) - 1) < 1e-6
    assert r['model'] == SVM and not at.exception


def test_muril_selected_without_artifacts_no_prediction_no_fallback(no_muril):
    at = run_app()
    at.radio(key='model_choice').set_value(MURIL).run()
    t = text_of(at)
    assert 'not available' in t and 'colab_muril_finetune.ipynb' in t     # setup message with no click needed
    assert at.radio(key='model_choice').value == MURIL                     # selection not changed by the app
    at = analyse(at, 'You are an idiot')
    assert at.session_state['result'] is None                              # nothing stored, nothing predicted
    assert not any('class="verdict"' in m.value for m in at.markdown)      # no verdict shown
    assert at.radio(key='model_choice').value == MURIL                     # still no silent switch


def test_switching_models_never_shows_stale_result(no_muril):
    at = run_app()
    at = analyse(at, 'Have a lovely day')
    assert any('class="verdict"' in m.value for m in at.markdown)
    at.radio(key='model_choice').set_value(MURIL).run()
    assert not any('class="verdict"' in m.value for m in at.markdown)
    at.radio(key='model_choice').set_value(SVM).run()                      # back: same text+model -> fresh again is allowed
    at.text_area(key='input_text').set_value('different text').run()
    assert not any('class="verdict"' in m.value for m in at.markdown)      # changed input -> hidden


def test_clear_button_resets():
    at = analyse(run_app(), 'some words here')
    at.button(key='reset').click().run()
    assert at.text_area(key='input_text').value == '' and at.session_state['result'] is None


def test_incomplete_artifacts_clear_message_not_crash(monkeypatch):
    monkeypatch.setattr(muril_model, 'check_artifacts', lambda d: (False, 'Incomplete model folder: Missing or empty: vocab.txt', None))
    at = run_app()
    assert 'unavailable' in options(at)[1]
    at.radio(key='model_choice').set_value(MURIL).run()
    assert not at.exception and 'vocab.txt' in text_of(at)


def test_artifacts_ok_but_torch_missing_is_reported(monkeypatch):
    monkeypatch.setattr(muril_model, 'check_artifacts', lambda d: (True, 'ok', {'max_len': 128}))
    for m in ('torch', 'transformers'):
        monkeypatch.setitem(sys.modules, m, None)                          # import raises ImportError
    at = run_app()
    at.radio(key='model_choice').set_value(MURIL).run()
    assert not at.exception and 'requirements-muril.txt' in text_of(at)


def test_wiring_with_stub_model_class_order_and_loader_error(monkeypatch):
    """Wiring only: a stub stands in for MuRIL. This is NOT evidence that a real model works."""
    monkeypatch.setattr(muril_model, 'check_artifacts', lambda d: (True, 'ok', {'max_len': 64}))
    for m in ('torch', 'transformers'):
        monkeypatch.setitem(sys.modules, m, types.ModuleType(m))
    calls = {}

    def fake_proba_fn(texts):
        calls['n'] = calls.get('n', 0) + 1
        return np.array([[0.1, 0.7, 0.2] if 'idiot' in t else [0.1, 0.1, 0.8] for t in texts])

    monkeypatch.setattr(muril_model, 'load', lambda d: ('tok', 'model', 'torch'))
    monkeypatch.setattr(muril_model, 'make_proba_fn', lambda tok, model, torch, max_len: fake_proba_fn)
    at = run_app()
    assert options(at)[1] == MURIL                                          # no "unavailable" suffix
    at.radio(key='model_choice').set_value(MURIL).run()
    at = analyse(at, 'you idiot')
    r = at.session_state['result']
    assert r['status'] == 'ok' and r['model'] == MURIL and r['pred'] == 1 and r['probs'] == [0.1, 0.7, 0.2]

    def boom(d):
        raise OSError('corrupt weights')
    monkeypatch.setattr(muril_model, 'load', boom)
    monkeypatch.setattr(muril_model, 'artifact_signature', lambda d: ('changed',))   # bust the resource cache
    at = analyse(run_app(), 'you idiot')
    at.radio(key='model_choice').set_value(MURIL).run()
    at = analyse(at, 'you idiot again')
    assert not at.exception and at.session_state['result']['status'] == 'error'
    assert 'corrupt weights' in text_of(at) or any('corrupt weights' in c.value for c in at.code)
