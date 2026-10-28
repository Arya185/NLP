"""Model-agnostic word-level explanation via leave-one-word-out occlusion."""


def word_importance(proba_fn, cleaned_text, class_id):
    """Drop in P(class_id) when each word is removed. Positive = word supports the class.

    proba_fn: callable taking a list of strings and returning an array (n, 3) of probabilities.
    A fitted sklearn Pipeline is accepted too (its predict_proba is used).
    """
    fn = getattr(proba_fn, 'predict_proba', proba_fn)
    words = cleaned_text.split()
    if not words:
        return []
    variants = [cleaned_text] + [' '.join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = fn(variants)[:, class_id]
    return [(w, float(probs[0] - p)) for w, p in zip(words, probs[1:])]
