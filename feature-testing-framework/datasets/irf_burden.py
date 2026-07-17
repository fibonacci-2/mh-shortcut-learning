"""Loader for the IRF dataset (Garg et al., ACL 2023) -- Perceived
Burdensomeness label. See irf_belong.py for the sibling loader (Thwarted
Belongingness) and the shared caveats (no user_id, 'control' is same-population
not neutral-population).

    python extract_gender.py --input data/raw/irf/irf-raw.csv \
        --out-dir data/gender --user-col user_id --text-col text
    python run.py --dataset irf_burden --data data/gender/irf-raw.csv
"""
import pandas as pd


def load(path='data/gender/irf-raw.csv'):
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    df['label'] = df['burden'].map({1: 'perceived_burdensomeness', 0: 'control'})
    return df


def conditions(df):
    return ['perceived_burdensomeness']
