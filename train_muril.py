"""Fine-tune google/muril-base-cased for 3-class hate speech detection (HATE/OFFENSIVE/NORMAL).

Run this on a GPU (Google Colab T4). It refuses to run on CPU unless --allow-cpu is given.
Uses the same src/preprocess.py (clean_text, label maps) and src/data.py splits as the SVM
baseline, so train/val/test never overlap on normalised text.

    python train_muril.py --out models/muril_final_model [--hasoc path.csv ...]
"""
import argparse
import hashlib
import inspect
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))

from data import build_splits, load_hasoc_csv  # noqa: E402
from preprocess import ID_TO_LABEL, LABEL_TO_ID, TARGET_NAMES  # noqa: E402

MIN_EVAL_N = 30        # below this many test rows per language, metrics are not meaningful
MIN_CLASS_N = 10       # ... and below this many per class
MIN_HINGLISH_TRAIN = 300   # below this, Hinglish capability is not claimed
WEIGHT_IF_RATIO_ABOVE = 1.25   # use class weights only when max/min class count exceeds this


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='google/muril-base-cased')
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'muril_final_model'),
                    help='must be <project>/models/muril_final_model for the Streamlit app to find it')
    ap.add_argument('--hasoc', nargs='*', default=[], help='user-downloaded HASOC CSVs (task_2 labels)')
    ap.add_argument('--hasoc-label-col', default='task_2')
    ap.add_argument('--epochs', type=float, default=4)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--allow-cpu', action='store_true')
    return ap.parse_args()


def warmup_steps_for(n_train, batch_size, epochs, fraction=0.1):
    """Warmup as an explicit step count. `warmup_ratio` is deprecated/removed in newer transformers."""
    import math
    return int(fraction * math.ceil(n_train / batch_size) * epochs)


def _jsonable(o):
    """json.dump fallback for numpy scalars/arrays."""
    if hasattr(o, 'tolist'):
        return o.tolist()
    raise TypeError(f'not JSON serialisable: {type(o).__name__}')


def confusion(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def evaluate_subset(name, df, preds):
    from sklearn.metrics import classification_report, f1_score
    n = len(df)
    per_class = df['label'].value_counts().reindex([0, 1, 2], fill_value=0)
    reliable = n >= MIN_EVAL_N and int(per_class.min()) >= MIN_CLASS_N
    print(f'\n=== {name}: n={n}, per-class={per_class.to_dict()} ===')
    if not reliable:
        print(f'WARNING: too few test examples for {name} (need >= {MIN_EVAL_N} total and '
              f'>= {MIN_CLASS_N} per class). The numbers below are NOT reliable evidence.')
    y, p = df['label'].values, np.asarray(preds)
    print(classification_report(y, p, labels=[0, 1, 2], target_names=TARGET_NAMES, digits=4, zero_division=0))
    cm = confusion(y, p)
    print('Confusion matrix (rows=true, cols=pred; order HATE, OFFENSIVE, NORMAL):')
    print(cm)
    rep = classification_report(y, p, labels=[0, 1, 2], target_names=TARGET_NAMES,
                                output_dict=True, zero_division=0)
    return {'n': n, 'reliable': reliable, 'macro_f1': float(f1_score(y, p, labels=[0, 1, 2], average='macro', zero_division=0)),
            'report': rep, 'confusion_matrix': cm.tolist()}


def main():
    args = parse_args()
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit('No GPU detected. In Colab: Runtime > Change runtime type > T4 GPU. '
                 'Pass --allow-cpu to override (very slow).')
    set_seed(args.seed)

    extra = [load_hasoc_csv(p, label_col=args.hasoc_label_col) for p in args.hasoc]
    for p, df in zip(args.hasoc, extra):
        print(f'{p}: kept {len(df)} rows, dropped {df.attrs["dropped_unmapped"]} with unmapped labels')
    stats = {}
    train, val, test = build_splits(stats, extra_hinglish=extra)
    print('cleaning/dedup counts:', stats)
    for nm, df in (('train', train), ('val', val), ('test', test)):
        print(nm, len(df), df.groupby(['language', 'label']).size().to_dict())
    for a, b, nm in ((train, val, 'train/val'), (train, test, 'train/test'), (val, test, 'val/test')):
        assert not set(a['clean_text']) & set(b['clean_text']), f'{nm} overlap on normalised text'
    n_hi_train = int((train['language'] == 'hinglish').sum())
    hinglish_enough = n_hi_train >= MIN_HINGLISH_TRAIN
    if not hinglish_enough:
        print('\n' + '!' * 78 + f'\nWARNING: only {n_hi_train} Hinglish training rows (< {MIN_HINGLISH_TRAIN}). '
              'This model cannot be expected to detect Hinglish hate speech reliably.\n'
              'Any Hinglish numbers below are anecdotal. English results are the meaningful ones.\n' + '!' * 78 + '\n')

    counts = train['label'].value_counts().reindex([0, 1, 2], fill_value=0)
    ratio = counts.max() / max(counts.min(), 1)
    use_weights = ratio > WEIGHT_IF_RATIO_ABOVE
    weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=train['label'].values) if use_weights else None
    print(f'class counts {counts.to_dict()} ratio={ratio:.2f} -> class weights: '
          f'{"ON " + str(np.round(weights, 3).tolist()) if use_weights else "OFF (classes balanced enough)"}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class DS(Dataset):
        def __init__(self, df):
            self.enc = tokenizer(df['clean_text'].tolist(), truncation=True, max_length=args.max_len)
            self.labels = df['label'].tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item['labels'] = self.labels[i]
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, id2label=ID_TO_LABEL, label2id=LABEL_TO_ID)
    w_tensor = torch.tensor(weights, dtype=torch.float) if use_weights else None

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            w = w_tensor.to(outputs.logits.device) if w_tensor is not None else None
            loss = nn.CrossEntropyLoss(weight=w)(outputs.logits.view(-1, 3), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {'macro_f1': f1_score(labels, np.argmax(logits, axis=1), average='macro')}

    # `evaluation_strategy` was renamed `eval_strategy` in newer transformers releases.
    strat_key = 'eval_strategy' if 'eval_strategy' in inspect.signature(TrainingArguments.__init__).parameters \
        else 'evaluation_strategy'
    targs = TrainingArguments(
        output_dir=os.path.join(args.out, 'checkpoints'), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64,
        learning_rate=args.lr, weight_decay=0.01,
        warmup_steps=warmup_steps_for(len(train), args.batch_size, args.epochs),
        save_strategy='epoch', load_best_model_at_end=True, metric_for_best_model='macro_f1',
        greater_is_better=True, save_total_limit=2, fp16=torch.cuda.is_available(),
        report_to='none', seed=args.seed, **{strat_key: 'epoch'})
    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=DS(train), eval_dataset=DS(val),
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])
    trainer.train()
    # Selection used validation only. Record the chosen checkpoint's validation score.
    val_metrics = trainer.evaluate()
    print('Best checkpoint validation metrics:', val_metrics)

    # Held-out evaluation (test split never used for training, tuning or checkpoint selection)
    test_ds = DS(test)
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    results = {'validation_best_checkpoint': {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
               'combined': evaluate_subset('Combined test', test, preds)}
    for lang in sorted(test['language'].unique()):
        mask = (test['language'] == lang).values
        results[lang] = evaluate_subset(f'{lang} test', test[mask], preds[mask])

    # Save model + tokenizer + label/config together
    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(HERE, 'src', 'preprocess.py'), 'rb') as f:
        pre_hash = hashlib.sha256(f.read()).hexdigest()
    config = {
        'base_model': args.model, 'id2label': {str(k): v for k, v in ID_TO_LABEL.items()},
        'label2id': LABEL_TO_ID, 'target_names': TARGET_NAMES, 'max_len': args.max_len,
        'preprocessing': 'src/preprocess.py clean_text (URLs removed before punctuation)',
        'preprocess_sha256': pre_hash, 'class_weights': None if weights is None else weights.tolist(),
        'train_counts': {str(k): int(v) for k, v in counts.items()},
        'hinglish_sources': ['seed (15 synthetic sentences)'] + list(args.hasoc),
        'hinglish_train_rows': n_hi_train,
        'english_train_rows': int((train['language'] == 'english').sum()),
        'limitations': None if hinglish_enough else (
            f'Only {n_hi_train} Hinglish training rows (synthetic seed). Hinglish hate-speech detection is NOT '
            'validated; do not rely on Hinglish predictions.'),
        'split_sizes': {'train': len(train), 'val': len(val), 'test': len(test)},
        'hyperparameters': {k: getattr(args, k) for k in ('epochs', 'batch_size', 'lr', 'patience', 'seed')},
    }
    with open(os.path.join(args.out, 'label_config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=_jsonable)
    with open(os.path.join(args.out, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=_jsonable)
    print(f'\nSaved model, tokenizer, label_config.json and eval_results.json to {args.out}')


if __name__ == '__main__':
    main()
