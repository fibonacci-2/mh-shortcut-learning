"""Loader for the UMD Reddit Suicidality Dataset (Shing et al. 2018), expert-annotated split.

No LIWC annotation is shipped with this corpus -- only the 6 pure-text
features in `features.py` (hedge_density, fp_singular, fp_plural,
excl_intensifiers, question_density, apology_selfblame) will be populated
unless you separately run LIWC on the text and merge those columns in.

Build the raw combined CSV once (post_title + post_body -> text, joined to
expert.csv's risk labels), then run gender extraction on it:

    python extract_gender.py --input data/raw/umd/umd-raw.csv \
        --out-dir data/gender --user-col user_id --text-col text
    python run.py --dataset umd --data data/gender/umd-raw.csv

Caveat: the control group (label NaN in expert.csv, i.e. non-mental-health-
subreddit users) has almost no self-disclosure -- only ~5/245 control users
state age+gender at all, vs. ~90/245 among the risk-labeled users. Any
condition-vs-control comparison here draws its control side from a
handful of users' repeated posts, not an independent sample -- treat
results as a pilot signal, not a confirmatory one.
"""
import pandas as pd

LABEL_NAMES = {
    'a': 'no_risk',
    'b': 'low_risk',
    'c': 'moderate_risk',
    'd': 'severe_risk',
}


def load(path='data/gender/umd-raw.csv'):
    """Returns a DataFrame with columns: text, label (risk level name,
    'control' for users with no risk annotation), gender ('male'/'female')."""
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    df['label'] = df['label'].map(LABEL_NAMES).fillna('control')
    return df


def conditions(df):
    return [c for c in LABEL_NAMES.values() if c in df['label'].unique()]
