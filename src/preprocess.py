"""Text preprocessing and label mappings, copied from colab_training_pipeline.ipynb."""
import re

import emoji

LABEL_TO_ID = {'HATE': 0, 'OFFENSIVE': 1, 'NORMAL': 2}
ID_TO_LABEL = {0: 'HATE', 1: 'OFFENSIVE', 2: 'NORMAL'}
TARGET_NAMES = ['HATE', 'OFFENSIVE', 'NORMAL']

HINGLISH_MAP = {
    'kyaa': 'kya', 'kia': 'kya', 'kyun': 'kyun', 'nahin': 'nahi', 'nai': 'nahi', 'nhn': 'nahi', 'ni': 'nahi',
    'haan': 'ha', 'haa': 'ha', 'hai': 'hai', 'he': 'hai', 'hain': 'hai', 'raha': 'raha', 'rha': 'raha',
    'karna': 'karna', 'krna': 'karna', 'bhai': 'bhai', 'bhaiya': 'bhai', 'bro': 'bhai', 'yaar': 'yaar',
    'kutta': 'kutte', 'kuttey': 'kutte', 'kamina': 'kamine', 'saala': 'saale', 'sala': 'saale',
    'chutiya': 'chutiya', 'chutiye': 'chutiya', 'madarchod': 'madarchod', 'mc': 'madarchod',
    'bhenchod': 'bhenchod', 'bc': 'bhenchod', 'bhosdike': 'bhosdike', 'bsdk': 'bhosdike', 'gandu': 'gandu'
}


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    # v2: strip URLs/mentions BEFORE the ':' replacement below, which otherwise turns
    # "https://x.com" into "https x com" (v1 / the Colab notebook did it in the wrong order).
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'\brt\b', ' ', text)
    text = emoji.demojize(text, delimiters=(' ', ' ')).replace(':', ' ')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [HINGLISH_MAP.get(tok, tok) for tok in text.split()]
    return re.sub(r'\s+', ' ', ' '.join(tokens)).strip()
