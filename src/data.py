"""Dataset construction and leakage-free splits.

English: Davidson labeled_data.csv (downloaded), up to 1000 rows per class, as in the notebook.
Hinglish: the notebook's 15 hand-written sentences. The notebook repeated them x80 before
splitting, so identical sentences landed in train/val/test. Here each unique sentence is kept
once and splits are made on unique cleaned text only.
"""
import os
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import clean_text

DAVIDSON_URL = ('https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/'
                'master/data/labeled_data.csv')

HINGLISH_SEED = [
    ('in sab ghuspaithiye logo ko desh se bahar nikalo warna ye sab khatam kar denge', 0),
    ('ye log desh ke gaddar hai inko goli maaro saalo ko', 0),
    ('all people of this religion are terrorists and deserve no human rights', 0),
    ('in minority walo ko sabak sikhana padega ye log virus ki tarah fail rahe hai', 0),
    ('inke religious places ko tod dalo saale jihadi log', 0),
    ('abe saale chutiye apna kaam kar zyada gyan mat pel', 1),
    ('tu bilkul gadha hai dimag ghutne me hai kya tera', 1),
    ('teri aukaat nahi hai mujhse baat karne ki chup baith bhenchod', 1),
    ('what a madarchod referee cheating karwa raha tha match me', 1),
    ('bhosdike ke chup kar warna thappad marunga', 1),
    ('bhai kal cricket match dekha kya kya zabardast six mara tha virat ne', 2),
    ('aaj ka mausam bohot accha hai chalo chai peene chalte hai', 2),
    ('bhai tu kaisa hai bohot time baad message kiya tune', 2),
    ('weekend pe goa chalne ka plan banate hai sab milke', 2),
    ('diwali ki shopping ke liye market me bahut mast lights lagi hai', 2),
]


def load_english(raw_dir='data/raw'):
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, 'davidson_raw.csv')
    if not os.path.exists(path):
        urllib.request.urlretrieve(DAVIDSON_URL, path)
    raw = pd.read_csv(path)
    df = pd.DataFrame({'text': raw['tweet'], 'label': raw['class'].astype(int), 'language': 'english'})
    parts = [g.sample(n=min(len(g), 1000), random_state=42) for _, g in df.groupby('label')]
    return pd.concat(parts, ignore_index=True)


def load_hinglish():
    return pd.DataFrame({'text': [s for s, _ in HINGLISH_SEED],
                         'label': [l for _, l in HINGLISH_SEED],
                         'language': 'hinglish'})


# HASOC-style fine-grained labels -> project ids. Anything not listed is DROPPED, never defaulted
# to NORMAL. HASOC definitions: HATE = hate speech; OFFN = offensive; PRFN = profane; NONE = neither.
# Profane-only content is mapped to OFFENSIVE (closest class); this mapping is a judgement call.
HASOC_LABELS = {'HATE': 0, 'OFFN': 1, 'PRFN': 1, 'NONE': 2}


def load_hasoc_csv(path, language='hinglish', text_col='text', label_col='task_2', sep=None):
    """Load a user-downloaded HASOC file (registration required, not redistributed here).

    Rows whose label is not in HASOC_LABELS (e.g. binary-only HOF/NOT, missing, unknown) are
    dropped and counted; nothing is silently treated as NORMAL.
    """
    raw = pd.read_csv(path, sep=sep, engine='python')
    for col in (text_col, label_col):
        if col not in raw.columns:
            raise ValueError(f'{path}: column {col!r} not found; columns are {list(raw.columns)}')
    lab = raw[label_col].astype(str).str.strip().str.upper().map(HASOC_LABELS)
    out = pd.DataFrame({'text': raw[text_col], 'label': lab, 'language': language})
    dropped = int(out['label'].isna().sum())
    out = out.dropna(subset=['label']).astype({'label': int})
    out.attrs['dropped_unmapped'] = dropped
    return out


def _prep(df, stats=None):
    df = df.copy()
    n0 = len(df)
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 2]
    n1 = len(df)
    # identical cleaned text must live in exactly one split; conflicting labels are dropped entirely
    conflict = df.groupby('clean_text')['label'].nunique()
    conflict = set(conflict[conflict > 1].index)
    df = df[~df['clean_text'].isin(conflict)]
    n2 = len(df)
    df = df.drop_duplicates(subset='clean_text').reset_index(drop=True)
    if stats is not None:
        lang = df['language'].iloc[0] if len(df) else '?'
        stats[lang] = {'input_rows': n0, 'too_short_removed': n0 - n1,
                       'label_conflict_removed': n1 - n2, 'duplicates_removed': n2 - len(df),
                       'unique_kept': len(df)}
    return df


def _split_80_10_10(df):
    tr, tmp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    va, te = train_test_split(tmp, test_size=0.5, random_state=42, stratify=tmp['label'])
    return tr, va, te


def build_splits(stats=None, extra_hinglish=None):
    """Return train, val, test (combined). `stats` (dict) receives per-language dedup counts.

    extra_hinglish: optional list of DataFrames (text, label, language) from real, licensed data,
    e.g. load_hasoc_csv(...). They are pooled with the seed, then deduplicated on cleaned text.
    """
    en = _prep(load_english(), stats)
    hi = _prep(pd.concat([load_hinglish()] + list(extra_hinglish or []), ignore_index=True), stats)
    if len(hi) < 150:
        # Tiny pool (the 15-sentence seed): 3 train / 1 val / 1 test per class.
        hi_tr, hi_tmp = train_test_split(hi, test_size=6, random_state=42, stratify=hi['label'])
        hi_va, hi_te = train_test_split(hi_tmp, test_size=3, random_state=42, stratify=hi_tmp['label'])
    else:
        hi_tr, hi_va, hi_te = _split_80_10_10(hi)
    en_tr, en_va, en_te = _split_80_10_10(en)

    def mix(a, b):
        return pd.concat([a, b]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    train, val, test = mix(en_tr, hi_tr), mix(en_va, hi_va), mix(en_te, hi_te)
    # Cross-language / cross-split safety net
    for name, other in (('val', val), ('test', test)):
        assert not set(train['clean_text']) & set(other['clean_text']), f'train/{name} overlap'
    assert not set(val['clean_text']) & set(test['clean_text']), 'val/test overlap'
    return train, val, test
