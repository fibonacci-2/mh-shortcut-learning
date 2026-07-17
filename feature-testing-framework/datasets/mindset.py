"""Loader for the Mindset Reddit mental-health corpus (LIWC-22 annotated CSV)."""
import pandas as pd

LIWC_COLS = [
    'WC', 'Affect', 'feeling', 'Social', 'family', 'friend', 'certitude',
    'emo_neg', 'swear', 'emo_anger', 'health', 'Physical', 'cogproc',
    'Analytic', 'Clout',
]


def load(path='data/strong-ann.csv'):
    """Returns a DataFrame with columns: text, label (condition name, 'control'
    for the control group), gender ('male'/'female'), plus LIWC columns."""
    df = pd.read_csv(path, low_memory=False)
    df['text'] = df['text'].fillna('')
    df['label'] = df['TID'].str.split('|').str[1]
    df['gender'] = df['gender_label'] if 'gender_label' in df.columns else df['gender']
    for c in LIWC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def conditions(df):
    return sorted(c for c in df['label'].dropna().unique() if c != 'control')
