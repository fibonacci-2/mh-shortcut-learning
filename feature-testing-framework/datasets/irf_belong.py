"""Loader for the IRF dataset (Garg et al., ACL 2023) -- Reddit posts labeled
for Thwarted Belongingness, one of the two Interpersonal Theory of Suicide
factors this corpus annotates (see irf_burden.py for the other, Perceived
Burdensomeness -- the two labels are independent, not mutually exclusive, so
each gets its own loader/condition rather than sharing one 'label' column).

No user_id in the source data (each row is a standalone post), so
extract_gender.py must be run with --user-col user_id --text-col text on the
prepared raw CSV, where user_id is just the row index (one "user" per post).
'control' here means "post from the same population without this factor
annotated," not a neutral healthy control -- same caveat as umd_demographics'
no_risk class.

    python extract_gender.py --input data/raw/irf/irf-raw.csv \
        --out-dir data/gender --user-col user_id --text-col text
    python run.py --dataset irf_belong --data data/gender/irf-raw.csv
"""
import pandas as pd


def load(path='data/gender/irf-raw.csv'):
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    df['label'] = df['belong'].map({1: 'thwarted_belongingness', 0: 'control'})
    return df


def conditions(df):
    return ['thwarted_belongingness']
