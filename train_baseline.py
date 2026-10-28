"""Train the notebook's dual TF-IDF + calibrated Linear SVM baseline and save it."""
import os
import sys

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from data import build_splits  # noqa: E402
from preprocess import ID_TO_LABEL, TARGET_NAMES  # noqa: E402

MODEL_PATH = 'models/svm_pipeline.joblib'


def build_pipeline():
    features = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ('char', TfidfVectorizer(ngram_range=(3, 5), max_features=15000, analyzer='char', sublinear_tf=True)),
    ])
    return Pipeline([
        ('features', features),
        ('clf', CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42), cv=3)),
    ])


def report(pipe, df, title):
    preds = pipe.predict(df['clean_text'])
    print(f'--- {title} (n={len(df)}) ---')
    print(classification_report(df['label'], preds, labels=[0, 1, 2], target_names=TARGET_NAMES,
                                digits=4, zero_division=0))


def main():
    stats = {}
    train, val, test = build_splits(stats)
    print('cleaning/dedup counts per language:', stats)
    for name, df in (('train', train), ('val', val), ('test', test)):
        counts = df.groupby(['language', 'label']).size().unstack(fill_value=0)
        counts.columns = [ID_TO_LABEL[c] for c in counts.columns]
        print(f'{name}: n={len(df)} (no cross-split overlap)\n{counts}')
    pipe = build_pipeline()
    pipe.fit(train['clean_text'], train['label'])

    report(pipe, val, 'Validation, combined')
    report(pipe, test[test['language'] == 'english'], 'Test, English')
    hi_test = test[test['language'] == 'hinglish']
    report(pipe, hi_test, 'Test, Hinglish - only 3 sentences, NOT a reliable estimate')

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f'saved {MODEL_PATH}')


if __name__ == '__main__':
    main()
