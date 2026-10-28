import html
import os
import sys

import joblib
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import muril_model  # noqa: E402
from explain import word_importance  # noqa: E402
from preprocess import ID_TO_LABEL, TARGET_NAMES, clean_text  # noqa: E402

SVM_PATH = os.path.join(ROOT, 'models', 'svm_pipeline.joblib')
MURIL_DIR = os.path.join(ROOT, 'models', 'muril_final_model')
PREPROCESS_PATH = os.path.join(ROOT, 'src', 'preprocess.py')
SVM_NAME = 'TF-IDF + Linear SVM'
MURIL_NAME = 'MuRIL Fine-Tuned'
MAX_CHARS = 2000
CLASS_CSS = {'HATE': 'hate', 'OFFENSIVE': 'off', 'NORMAL': 'ok'}

st.set_page_config(page_title='Hate Speech Detection', page_icon='🛡️', layout='centered')

CSS = """
<style>
:root{
  --ivory:#efe9d9; --ivory-hi:#faf6ea; --ivory-lo:#ddd5bf; --shell:#d9d3c3;
  --ink:#26302b; --ink-soft:#6b6552; --edge:#b9b09a; --edge-dark:#9d947c;
  --green:#2f5d50; --green-hi:#3f7566; --green-lo:#1f4238;
  --hate:#9b3b2e; --off:#b7791f; --ok:#2f6b4f;
}
html, body, .stApp{ background:linear-gradient(180deg,#dcd6c6 0%,#cfc8b6 100%) !important; color:var(--ink); }
.stApp, .stApp *{ font-family:"Avenir Next","Helvetica Neue",Helvetica,Arial,sans-serif; }
[data-testid="stIconMaterial"], .stApp [class*="material-icons"]{ font-family:"Material Symbols Rounded","Material Icons" !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{ display:none !important; }
header[data-testid="stHeader"]{ background:transparent !important; }
.block-container{ max-width:820px !important; padding-top:2.2rem !important; padding-bottom:3rem !important; }

/* engraved small labels */
.eng{ font-family:ui-monospace,"SF Mono",Menlo,monospace !important; font-size:.68rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-soft); text-shadow:0 1px 0 rgba(255,255,255,.75); margin:.1rem 0 .45rem; }
.rule{ height:0; border-top:1px solid #c2b9a1; border-bottom:1px solid #fffdf3; margin:1.1rem 0 1rem; }

/* header */
.hdr{ display:flex; align-items:center; gap:14px; margin-bottom:1.1rem; }
.hdr .mark{ width:46px; height:46px; border-radius:12px; display:grid; place-items:center;
  background:linear-gradient(180deg,var(--ivory-hi),var(--ivory-lo)); border:1px solid var(--edge);
  box-shadow:0 2px 0 #b3aa92, 0 5px 10px rgba(70,60,40,.25), inset 0 1px 0 #fff; }
.hdr h1{ font-size:1.55rem; line-height:1.15; margin:0; padding:0; color:var(--ink); font-weight:700; letter-spacing:.01em;
  text-shadow:0 1px 0 rgba(255,255,255,.8); }
.hdr .sub{ font-size:.86rem; color:var(--ink-soft); margin-top:2px; }
.hdr .grow{ flex:1; }
.pill{ display:inline-flex; align-items:center; gap:7px; padding:6px 12px; border-radius:999px; font-size:.72rem;
  font-family:ui-monospace,"SF Mono",Menlo,monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--ink);
  background:#e3dcc8; border:1px solid var(--edge); box-shadow:inset 0 2px 4px rgba(60,50,30,.25), 0 1px 0 #fff; }

/* lamps */
.lamp{ width:10px; height:10px; border-radius:50%; display:inline-block; flex:none;
  box-shadow:inset 0 1px 2px rgba(0,0,0,.45), 0 0 0 2px #cfc7b0, 0 1px 0 2px #fffdf3; }
.lamp.on{ background:radial-gradient(circle at 35% 30%,#9fd8bd,#2f8f6a 60%,#1f6a4d); }
.lamp.off{ background:radial-gradient(circle at 35% 30%,#d8d2c2,#9d947c 70%); }
.lamp.warn{ background:radial-gradient(circle at 35% 30%,#f2d38a,#c8892b 65%,#8e5c14); }
.lamp.hate{ background:radial-gradient(circle at 35% 30%,#e9a79b,#b04535 60%,#7c2a1f); }
.lamp.okc{ background:radial-gradient(circle at 35% 30%,#9fd8bd,#2f8f6a 60%,#1f6a4d); }
.lamp.big{ width:16px; height:16px; }

/* main panel */
.st-key-panel, .st-key-results{ position:relative; background:linear-gradient(180deg,var(--ivory-hi) 0%,var(--ivory) 14%,#e9e2cf 100%);
  border:1px solid var(--edge); border-radius:18px; padding:1.4rem 1.5rem 1.3rem !important; margin-bottom:1.3rem;
  box-shadow:inset 0 1px 0 #fff, inset 0 -2px 0 rgba(120,105,70,.15), 0 2px 0 #b3aa92, 0 14px 30px rgba(70,60,40,.25); }
.st-key-panel::before, .st-key-panel::after, .st-key-results::before, .st-key-results::after{
  content:""; position:absolute; top:11px; width:8px; height:8px; border-radius:50%;
  background:radial-gradient(circle at 35% 30%,#f4efe0,#a79e86); box-shadow:inset 0 0 0 1px rgba(90,80,55,.35), 0 1px 0 #fff; }
.st-key-panel::before, .st-key-results::before{ left:11px; } .st-key-panel::after, .st-key-results::after{ right:11px; }

/* model selector: physical switch plates */
.st-key-model_choice, .st-key-model_choice > div, .st-key-model_choice [data-testid="stRadio"]{ width:100% !important; }
.st-key-model_choice [role="radiogroup"]{ display:flex !important; gap:10px; flex-wrap:nowrap; width:100%; }
.st-key-model_choice [role="radiogroup"] > div{ flex:1 1 0; min-width:0; }
.st-key-model_choice [data-testid="stRadioOption"]{ display:flex; width:100%; white-space:nowrap; box-sizing:border-box; margin:0 !important; padding:11px 14px !important; border-radius:12px; cursor:pointer;
  justify-content:center; background:linear-gradient(180deg,var(--ivory-hi),var(--ivory-lo)); border:1px solid var(--edge);
  box-shadow:0 3px 0 #b3aa92, 0 6px 10px rgba(70,60,40,.2), inset 0 1px 0 #fff; transition:transform .06s, box-shadow .06s; }
.st-key-model_choice [data-testid="stRadioOption"] > div > div:first-child{ display:none !important; }
.st-key-model_choice [data-testid="stRadioOption"] p{ font-size:.86rem !important; font-weight:600; letter-spacing:.02em; color:var(--ink); }
.st-key-model_choice [data-testid="stRadioOption"][data-selected="true"]{ transform:translateY(2px);
  background:linear-gradient(180deg,#d3cbb4,#e1dac6); border-color:var(--edge-dark);
  box-shadow:inset 0 3px 7px rgba(60,50,30,.4), 0 1px 0 #fff; }
.st-key-model_choice [data-testid="stRadioOption"][data-selected="true"] p{ color:var(--green-lo); }
.model-note{ font-size:.8rem; color:var(--ink-soft); margin:.55rem 0 0; display:flex; align-items:center; gap:8px; }

/* recessed text area */
.st-key-input_text textarea, div[data-testid="stTextArea"] textarea{ background:#e4ddc9 !important; color:var(--ink) !important;
  font-size:1.02rem !important; line-height:1.5; border-radius:12px !important; border:1px solid var(--edge) !important;
  box-shadow:inset 0 4px 10px rgba(60,50,30,.38), inset 0 -1px 0 rgba(255,255,255,.7) !important; caret-color:var(--green); }
.st-key-input_text textarea::placeholder{ color:#8d8670 !important; opacity:1; }
div[data-testid="stTextArea"] > div, div[data-baseweb="textarea"], div[data-baseweb="base-input"]{ background:transparent !important;
  border:none !important; box-shadow:none !important; }
.st-key-input_text textarea:focus{ outline:2px solid rgba(47,93,80,.45) !important; outline-offset:1px; }
div[data-testid="stTextAreaCharacterCount"], [data-testid="InputInstructions"]{ color:var(--ink-soft) !important;
  font-family:ui-monospace,Menlo,monospace; font-size:.7rem; }

/* tactile buttons */
.st-key-analyse button, .st-key-reset button{ border-radius:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
  font-size:.8rem; padding:.7rem 1rem; transition:transform .06s, box-shadow .06s; }
.st-key-analyse button{ background:linear-gradient(180deg,var(--green-hi),var(--green)) !important; color:#f6f1e1 !important;
  border:1px solid var(--green-lo) !important; text-shadow:0 -1px 0 rgba(0,0,0,.35);
  box-shadow:0 4px 0 var(--green-lo), 0 9px 14px rgba(30,50,40,.35), inset 0 1px 0 rgba(255,255,255,.28) !important; }
.st-key-analyse button:hover{ filter:brightness(1.05); }
.st-key-reset button{ background:linear-gradient(180deg,var(--ivory-hi),var(--ivory-lo)) !important; color:var(--ink) !important;
  border:1px solid var(--edge) !important; box-shadow:0 3px 0 #b3aa92, 0 6px 10px rgba(70,60,40,.2), inset 0 1px 0 #fff !important; }
.st-key-analyse button:active{ transform:translateY(3px); box-shadow:0 1px 0 var(--green-lo), inset 0 2px 6px rgba(0,0,0,.35) !important; }
.st-key-reset button:active{ transform:translateY(2px); box-shadow:0 1px 0 #b3aa92, inset 0 2px 6px rgba(60,50,30,.35) !important; }

/* results */
.empty{ text-align:center; padding:1.6rem 0 1.2rem; }
.empty .ring{ width:54px; height:54px; margin:0 auto 12px; border-radius:50%; background:#e3dcc8; border:1px solid var(--edge);
  box-shadow:inset 0 3px 7px rgba(60,50,30,.35), 0 1px 0 #fff; display:grid; place-items:center; }
.empty .t{ font-size:1.05rem; font-weight:650; color:var(--ink); } .empty .s{ font-size:.86rem; color:var(--ink-soft); margin-top:3px; }
.verdict{ display:flex; align-items:center; gap:16px; padding:14px 16px; border-radius:14px; background:#e4ddc9;
  border:1px solid var(--edge); box-shadow:inset 0 3px 8px rgba(60,50,30,.3), 0 1px 0 #fff; }
.verdict .cls{ font-size:1.7rem; font-weight:800; letter-spacing:.06em; line-height:1.1; }
.verdict .cls.hate{ color:var(--hate); } .verdict .cls.off{ color:var(--off); } .verdict .cls.ok{ color:var(--ok); }
.verdict .meta{ font-size:.78rem; color:var(--ink-soft); margin-top:3px; }
.meter{ display:grid; grid-template-columns:96px 1fr 58px; align-items:center; gap:12px; margin:9px 0; }
.meter .lab{ font-family:ui-monospace,Menlo,monospace; font-size:.72rem; letter-spacing:.12em; color:var(--ink-soft); }
.meter.top .lab{ color:var(--ink); font-weight:700; }
.meter .trk{ height:14px; border-radius:8px; background:#e0d9c4; border:1px solid var(--edge);
  box-shadow:inset 0 2px 5px rgba(60,50,30,.4), 0 1px 0 #fff; overflow:hidden; }
.meter .fil{ height:100%; border-radius:7px; box-shadow:inset 0 1px 0 rgba(255,255,255,.4), inset 0 -1px 0 rgba(0,0,0,.2); }
.fil.hate{ background:linear-gradient(180deg,#c0584a,var(--hate)); } .fil.off{ background:linear-gradient(180deg,#d69a3c,var(--off)); }
.fil.ok{ background:linear-gradient(180deg,#4f9272,var(--ok)); }
.meter .val{ font-family:ui-monospace,Menlo,monospace; font-size:.8rem; text-align:right; color:var(--ink); }
.chips span{ display:inline-block; padding:3px 9px; margin:3px 3px 3px 0; border-radius:7px; font-size:.92rem; color:var(--ink);
  border:1px solid rgba(90,80,55,.25); box-shadow:inset 0 1px 0 rgba(255,255,255,.5); }
.legend{ font-size:.75rem; color:var(--ink-soft); margin-top:6px; }
.note{ display:flex; gap:9px; align-items:flex-start; font-size:.78rem; color:var(--ink-soft); padding:0 .3rem; margin-top:-.2rem; }
.note .lamp{ margin-top:4px; }
.hint{ font-size:.82rem; color:var(--ink-soft); margin:.1rem 0 0; }
[data-testid="stExpander"]{ border:1px solid var(--edge) !important; border-radius:12px !important; background:#e9e2cf; }
[data-testid="stExpander"] summary p{ font-size:.82rem; font-weight:600; color:var(--ink); }
[data-testid="stAlert"]{ border-radius:12px; }
@media (max-width:640px){ .meter{ grid-template-columns:78px 1fr 52px; } .hdr .pill{ display:none; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_svm():
    pipe = joblib.load(SVM_PATH)
    assert list(pipe.classes_) == [0, 1, 2], f'unexpected class order {pipe.classes_}'
    return pipe


@st.cache_resource
def load_muril(model_dir, signature, max_len):
    """Cached per artifact fingerprint: reruns reuse the model, replaced artifacts reload it."""
    tok, model, torch = muril_model.load(model_dir)
    return muril_model.make_proba_fn(tok, model, torch, max_len)


def muril_status():
    """(ready, message, config). Imports torch only if the artifacts look valid."""
    ok, msg, cfg = muril_model.check_artifacts(MURIL_DIR)
    if not ok:
        return False, msg, cfg
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        return False, (f'MuRIL artifacts found but a dependency is missing ({e.name}). '
                       'Run: .venv312/bin/pip install -r requirements-muril.txt'), cfg
    return True, 'ok', cfg


def reset_all():
    st.session_state['input_text'] = ''
    st.session_state['result'] = None


svm_ready = os.path.exists(SVM_PATH)
muril_ready, muril_msg, muril_cfg = muril_status()
st.session_state.setdefault('result', None)

# ---------------------------------------------------------------- header
SHIELD = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M12 2.5l8 3v6.2c0 4.7-3.2 8.3-8 9.8-4.8-1.5-8-5.1-8-9.8V5.5l8-3z" '
          'fill="#2f5d50" stroke="#1f4238" stroke-width="1"/><path d="M12 4.6l5.9 2.2v4.9c0 3.6-2.3 6.3-5.9 7.6V4.6z" fill="#3f7566"/>'
          '<path d="M8.6 12.2l2.5 2.5 4.4-4.9" stroke="#f6f1e1" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>')
n_up = int(svm_ready) + int(muril_ready)
st.markdown(
    f'<div class="hdr"><div class="mark">{SHIELD}</div><div><h1>Hate Speech Detection</h1>'
    '<div class="sub">English + Hinglish · Text Classification Workbench</div></div><div class="grow"></div>'
    f'<div class="pill"><span class="lamp {"on" if n_up else "off"}"></span>{n_up} of 2 models ready</div></div>',
    unsafe_allow_html=True)

# ---------------------------------------------------------------- input panel
with st.container(key='panel'):
    st.markdown('<div class="eng">Model select</div>', unsafe_allow_html=True)
    # Default (first option) is always the SVM baseline. Options are fixed values; only the display text
    # shows availability, so a deliberate selection is never changed by the app.
    availability = {SVM_NAME: svm_ready, MURIL_NAME: muril_ready}
    model = st.radio('Model', [SVM_NAME, MURIL_NAME], horizontal=True, key='model_choice', label_visibility='collapsed',
                     format_func=lambda m: m if availability[m] else f'{m} · unavailable')
    if model == SVM_NAME:
        lamp, note = ('on', 'Baseline classifier · calibrated probabilities') if svm_ready else \
            ('off', 'Unavailable: models/svm_pipeline.joblib not found. Run train_baseline.py.')
    else:
        lamp, note = ('on', 'Fine-tuned google/muril-base-cased') if muril_ready else \
            ('off', 'Unavailable: ' + html.escape(muril_msg))
    st.markdown(f'<div class="model-note"><span class="lamp {lamp}"></span><span>{note}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="rule"></div><div class="eng">Text input</div>', unsafe_allow_html=True)
    text = st.text_area('Text input', key='input_text', placeholder='Enter or paste text to analyse…', height=170,
                        max_chars=MAX_CHARS, label_visibility='collapsed')
    c1, c2 = st.columns([3, 1])
    analyse = c1.button('Analyse text', key='analyse', type='primary', width='stretch')
    c2.button('Clear', key='reset', width='stretch', on_click=reset_all)

# ---------------------------------------------------------------- analysis (only on explicit click)
model_ready = svm_ready if model == SVM_NAME else muril_ready
if analyse:
    cleaned = clean_text(text)
    res = {'text': text, 'model': model, 'cleaned': cleaned, 'status': 'ok'}
    if not text.strip():
        res['status'] = 'empty'
    elif not cleaned:
        res['status'] = 'blank_after_clean'
    elif not model_ready:
        res = None  # nothing to store: the results panel shows the unavailable message instead
    else:
        try:
            if model == SVM_NAME:
                proba_fn = load_svm().predict_proba
            else:
                proba_fn = load_muril(MURIL_DIR, muril_model.artifact_signature(MURIL_DIR),
                                      int(muril_cfg.get('max_len', muril_model.DEFAULT_MAX_LEN)))
                if muril_model.preprocess_matches(muril_cfg, PREPROCESS_PATH) is False:
                    res['preprocess_warning'] = True
            probs = proba_fn([cleaned])[0]
            pred = int(probs.argmax())
            res.update(probs=[float(p) for p in probs], pred=pred, scores=word_importance(proba_fn, cleaned, pred))
        except Exception as e:  # corrupt weights, bad tokenizer, out-of-memory, ...
            res.update(status='error', error=f'{type(e).__name__}: {e}')
    st.session_state['result'] = res

# ---------------------------------------------------------------- results
res = st.session_state['result']
fresh = res is not None and res['text'] == text and res['model'] == model

with st.container(key='results'):
    st.markdown('<div class="eng">Analysis output</div>', unsafe_allow_html=True)
    if not model_ready:
        # Selected model cannot run: say so immediately (no click needed) and never show another model's output.
        st.error(f'{model} is not available, so nothing can be predicted with it. No other model is substituted.')
        if model == MURIL_NAME:
            st.markdown('**How to make MuRIL available**\n\n'
                        '1. Run `colab_muril_finetune.ipynb` on a Colab **T4 GPU** (it trains and exports `muril_final_model.zip`).\n'
                        '2. Unzip it into `NLP/models/` so `models/muril_final_model/label_config.json` exists.\n'
                        '3. `.venv312/bin/pip install -r requirements-muril.txt`, then restart Streamlit.\n\n'
                        f'Reason: {muril_msg}')
        else:
            st.markdown('Reason: models/svm_pipeline.joblib not found. Run `.venv312/bin/python train_baseline.py`.')
        st.markdown('Select **TF-IDF + Linear SVM** above to keep analysing text.')
    elif not fresh:
        if res is not None:
            sub = 'Input or model changed since the last analysis. Press Analyse text to update.'
        elif not text.strip():
            sub = 'Enter some text above and press Analyse text.'
        else:
            sub = 'Press Analyse text to run the selected model.'
        st.markdown('<div class="empty"><div class="ring"><span class="lamp off"></span></div>'
                    f'<div class="t">Awaiting input</div><div class="s">Your analysis will appear here.<br>{sub}</div></div>',
                    unsafe_allow_html=True)
    elif res['status'] == 'empty':
        st.markdown('<div class="hint">Nothing to analyse. Enter or paste some text first.</div>', unsafe_allow_html=True)
    elif res['status'] == 'blank_after_clean':
        st.markdown('<div class="hint">Nothing is left after preprocessing (only links, mentions or symbols). '
                    'Enter some text.</div>', unsafe_allow_html=True)
    elif res['status'] == 'error':
        st.error(f'{model} failed while loading or predicting, so nothing was predicted. No other model was substituted.')
        st.code(res['error'], language=None)
    else:
        probs, pred = res['probs'], res['pred']
        label = ID_TO_LABEL[pred]
        css = CLASS_CSS[label]
        lamp_cls = {'hate': 'hate', 'off': 'warn', 'ok': 'okc'}[css]
        if res.get('preprocess_warning'):
            st.warning('src/preprocess.py differs from the version this MuRIL model was trained with.')
        st.markdown(f'<div class="verdict"><span class="lamp big {lamp_cls}"></span><div><div class="cls {css}">{label}</div>'
                    f'<div class="meta">{html.escape(res["model"])} · probability {probs[pred]:.1%}</div></div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="rule"></div><div class="eng">Class probabilities</div>', unsafe_allow_html=True)
        bars = ''
        for i, name in enumerate(TARGET_NAMES):
            bars += (f'<div class="meter{" top" if i == pred else ""}"><div class="lab">{name}</div>'
                     f'<div class="trk"><div class="fil {CLASS_CSS[name]}" style="width:{probs[i] * 100:.1f}%"></div></div>'
                     f'<div class="val">{probs[i]:.1%}</div></div>')
        st.markdown(bars, unsafe_allow_html=True)
        with st.expander('Preprocessed text'):
            st.code(res['cleaned'], language=None)
        scores = res['scores']
        st.markdown(f'<div class="rule"></div><div class="eng">Word contributions to {label}</div>', unsafe_allow_html=True)
        top = max((abs(s) for _, s in scores), default=0) or 1.0
        chips = ''
        for w, s in scores:
            a = min(1.0, abs(s) / top) * 0.75
            base = {'hate': '155,59,46', 'off': '183,121,31', 'ok': '47,107,79'}[css] if s > 0 else '84,98,108'
            col = f'rgba({base},{a:.2f})'
            chips += f'<span style="background:{col}" title="{s:+.3f}">{html.escape(w)}</span>'
        st.markdown(f'<div class="chips">{chips}</div><div class="legend">Coloured: removing the word lowers the '
                    f'{label} probability (supports it). Slate: removing it raises it. Method: leave-one-word-out '
                    'occlusion on the cleaned text.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- data-quality note
st.markdown('<div class="note"><span class="lamp warn"></span><span><b>Data note.</b> Hinglish training data is currently only a '
            'small synthetic seed (15 sentences), so Hinglish predictions are unreliable. English uses the Davidson tweet set.</span></div>',
            unsafe_allow_html=True)
