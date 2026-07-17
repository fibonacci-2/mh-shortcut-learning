"""Loader for a pre-processed UMD subset that already has LIWC-22 annotation
and self-disclosure gender extraction applied (same schema as Mindset's
strong-ann.csv, plus confidence/matched_excerpt/pattern_type) -- so unlike
umd.py / umd_crowd.py, this one does NOT need extract_gender.py run on it.

Binary risk labels only (`condition`: 'at_risk' / 'no_risk'). 'no_risk' is
used as the control class here, but note it's still a SuicideWatch-adjacent
user assessed as low risk by an annotator, not a neutral background-subreddit
control like Mindset's control group -- interpret condition-vs-control
effects accordingly.

Following the same "strong" convention as Mindset, we keep only
confidence == 'high' self-disclosures (explicit age+gender, e.g. "17M"),
dropping the medium/low-confidence contextual gender guesses this file also
carries.
"""
import pandas as pd


def load(path='data/raw/umd_demographics/umd_standardized_demographics.csv'):
    """Returns a DataFrame with columns: text, label ('at_risk'/'control'),
    gender ('male'/'female'), plus LIWC-22 columns."""
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    df = df[(df['confidence'] == 'high') & (df['gender'].isin(['male', 'female']))].copy()
    df['label'] = df['condition'].map({'at_risk': 'at_risk', 'no_risk': 'control'})
    return df


def conditions(df):
    return ['at_risk']
