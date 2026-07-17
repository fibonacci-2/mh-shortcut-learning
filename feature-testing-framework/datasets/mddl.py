"""Loader for the MDDL (Multimodal Depressive Dictionary Learning, Shen et al.
2017) Twitter dataset -- self-disclosure gender extraction only found a label
for 973/11,060 users (8.8%), similar low coverage to umd/umd_crowd (extraction
runs on bio + seed tweet + full timeline text, see mddl_prepare.py). No LIWC
annotation yet (not included in the original merge_liwc.py batch) -- only the
token-counted features in features.py will be live until it's run through
LIWC-22 too.

    python extract_gender.py --input data/raw/mddl/mddl-raw.csv \
        --out-dir data/gender --user-col user_id --text-col text
    python run.py --dataset mddl --data data/gender/mddl-raw.csv
"""
import pandas as pd


def load(path='data/gender/mddl-raw.csv'):
    """Returns a DataFrame with columns: text, label ('depression'/'control'),
    gender ('male'/'female')."""
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    return df


def conditions(df):
    return ['depression']
