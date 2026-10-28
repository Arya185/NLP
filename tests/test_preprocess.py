import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data import HINGLISH_SEED, _prep, load_hinglish  # noqa: E402
from preprocess import HINGLISH_MAP, ID_TO_LABEL, LABEL_TO_ID, TARGET_NAMES, clean_text  # noqa: E402


def test_label_maps():
    assert LABEL_TO_ID == {'HATE': 0, 'OFFENSIVE': 1, 'NORMAL': 2}
    assert ID_TO_LABEL == {0: 'HATE', 1: 'OFFENSIVE', 2: 'NORMAL'}
    assert TARGET_NAMES == [ID_TO_LABEL[i] for i in range(3)]
    assert all(ID_TO_LABEL[LABEL_TO_ID[k]] == k for k in LABEL_TO_ID)


def test_clean_text_notebook_sample():
    out = clean_text('Abe saalaaa kutteeee https://test.com @user kyaaa haaal hai 😡')
    assert '@' not in out and 'user' not in out.split()
    assert 'https' not in out and 'test' not in out.split() and 'com' not in out.split()
    assert 'kyaa' not in out.split()
    assert 'enraged' in out or 'angry' in out  # emoji expanded to words


def test_clean_text_hinglish_normalisation():
    assert clean_text('bsdk') == 'bhosdike'
    assert clean_text('MC BC') == 'madarchod bhenchod'
    assert clean_text('nhn') == 'nahi'
    assert clean_text('hellooooo') == 'helloo'


def test_clean_text_edge_cases():
    assert clean_text('') == ''
    assert clean_text('   ') == ''
    assert clean_text(None) == ''
    assert clean_text('RT @bob: hi!!!') == 'hi'


def test_url_removal_variants():
    assert clean_text('visit https://a.com/x?y=1 now') == 'visit now'
    assert clean_text('http://t.co/abc see') == 'see'
    assert clean_text('see www.foo.com!! ok') == 'see ok'
    assert clean_text('HTTPS://EXAMPLE.COM/Path hi') == 'hi'


def test_mentions_rt_and_emoji():
    assert clean_text('@bob_1 hi RT yes') == 'hi yes'
    assert clean_text('hi 😡') == 'hi enraged_face'
    assert clean_text('soooo gooood') == 'soo good'


def test_punctuation_and_idempotence():
    assert clean_text('Tu pagal hai, bsdk!!') == 'tu pagal hai bhosdike'
    for t in ['visit https://a.com now', 'Abe saalaaa 😡 @x', 'bsdk mc!!!']:
        assert clean_text(clean_text(t)) == clean_text(t)


def test_hinglish_map_values_are_clean():
    assert all(v == v.lower() for v in HINGLISH_MAP.values())


def test_hinglish_seed_unique_and_balanced():
    df = _prep(load_hinglish())
    assert len(df) == len(HINGLISH_SEED) == 15
    assert df['clean_text'].is_unique
    assert df['label'].value_counts().to_dict() == {0: 5, 1: 5, 2: 5}


def test_hasoc_mapping_drops_unmapped_never_defaults_to_normal(tmp_path):
    import pandas as pd
    from data import HASOC_LABELS, load_hasoc_csv
    assert HASOC_LABELS == {'HATE': 0, 'OFFN': 1, 'PRFN': 1, 'NONE': 2}
    f = tmp_path / 'h.csv'
    pd.DataFrame({'text': list('abcdef'), 'task_2': ['HATE', 'OFFN', 'prfn ', 'NONE', None, 'UNKNOWN']}).to_csv(f, index=False)
    out = load_hasoc_csv(str(f))
    assert out['label'].tolist() == [0, 1, 1, 2]
    assert out.attrs['dropped_unmapped'] == 2


def test_prep_removes_duplicates_and_label_conflicts():
    import pandas as pd
    df = pd.DataFrame({'text': ['Hello there friend', 'hello there friend!!', 'same text here', 'Same text here', 'unique one text'],
                       'label': [2, 2, 0, 1, 2], 'language': 'english'})
    stats = {}
    out = _prep(df, stats)
    assert sorted(out['clean_text']) == ['hello there friend', 'unique one text']
    assert stats['english']['label_conflict_removed'] == 2 and stats['english']['duplicates_removed'] == 1


def test_word_importance_with_stub_model():
    import numpy as np
    from explain import word_importance

    def stub(texts):  # "hate" word drives class 0
        p = np.array([[0.9, 0.05, 0.05] if 'hate' in t.split() else [0.1, 0.1, 0.8] for t in texts])
        return p
    res = dict(word_importance(stub, 'i hate you', 0))
    assert res['hate'] > 0.7 and res['i'] == 0 and res['you'] == 0
    assert word_importance(stub, '', 0) == []


def _write_artifacts(d, **over):
    """Create a structurally valid (fake, zero-weight-content) MuRIL folder for loader/validation tests only."""
    import json
    d.mkdir(exist_ok=True)
    id2 = {str(k): v for k, v in ID_TO_LABEL.items()}
    files = {
        'config.json': json.dumps({'model_type': 'bert', 'id2label': id2, 'num_labels': 3}),
        'model.safetensors': 'x', 'tokenizer_config.json': '{}', 'vocab.txt': '[PAD]',
        'label_config.json': json.dumps({'id2label': id2, 'label2id': LABEL_TO_ID, 'max_len': 128}),
        'eval_results.json': '{}',
    }
    files.update(over)
    for name, content in files.items():
        if content is not None:
            (d / name).write_text(content)
    return d


def test_muril_artifact_check_missing_folder_and_incomplete(tmp_path):
    import muril_model
    ok, msg, _ = muril_model.check_artifacts(str(tmp_path / 'nope'))
    assert not ok and 'not found' in msg
    for missing in ('config.json', 'model.safetensors', 'tokenizer_config.json', 'vocab.txt', 'label_config.json', 'eval_results.json'):
        d = _write_artifacts(tmp_path / f'no_{missing}', **{missing: None})
        ok, msg, _ = muril_model.check_artifacts(str(d))
        assert not ok and missing.split('.')[0] in msg, (missing, msg)
    d = _write_artifacts(tmp_path / 'empty_weights', **{'model.safetensors': ''})
    assert not muril_model.check_artifacts(str(d))[0]
    d = _write_artifacts(tmp_path / 'tok_json', **{'vocab.txt': None, 'tokenizer.json': '{}'})
    assert muril_model.check_artifacts(str(d))[0]


def test_muril_artifact_check_rejects_malformed_and_mismatched(tmp_path):
    import json
    import muril_model
    d = _write_artifacts(tmp_path / 'badjson', **{'label_config.json': '{not json'})
    ok, msg, _ = muril_model.check_artifacts(str(d))
    assert not ok and 'malformed' in msg
    swapped = {'id2label': {'0': 'NORMAL', '1': 'HATE', '2': 'OFFENSIVE'}, 'label2id': {'NORMAL': 0, 'HATE': 1, 'OFFENSIVE': 2}}
    d = _write_artifacts(tmp_path / 'swapped', **{'label_config.json': json.dumps(swapped)})
    ok, msg, _ = muril_model.check_artifacts(str(d))
    assert not ok and 'do not match' in msg
    d = _write_artifacts(tmp_path / 'hfswapped', **{'config.json': json.dumps({'id2label': swapped['id2label']})})
    ok, msg, _ = muril_model.check_artifacts(str(d))
    assert not ok and 'config.json id2label' in msg
    ok, msg, cfg = muril_model.check_artifacts(str(_write_artifacts(tmp_path / 'good')))
    assert ok and cfg['max_len'] == 128


def test_build_splits_accepts_extra_hinglish_without_overlap(monkeypatch):
    import pandas as pd
    import data
    en = pd.DataFrame({'text': [f'english sentence number {i} about topic {i % 7}' for i in range(300)],
                       'label': [i % 3 for i in range(300)], 'language': 'english'})
    extra = pd.DataFrame({'text': [f'yeh hinglish vakya sankhya {i} hai bhai' for i in range(300)],
                          'label': [i % 3 for i in range(300)], 'language': 'hinglish'})
    monkeypatch.setattr(data, 'load_english', lambda *a, **k: en)
    stats = {}
    tr, va, te = data.build_splits(stats, extra_hinglish=[extra])
    # clean_text collapses '111'->'11' etc., so a few synthetic strings legitimately dedupe
    assert 300 <= stats['hinglish']['unique_kept'] <= 315
    sets = [set(x['clean_text']) for x in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert (te['language'] == 'hinglish').sum() > 20
