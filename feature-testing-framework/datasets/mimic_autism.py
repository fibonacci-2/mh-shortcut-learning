"""Loader for MIMIC-IV autism ICD-coding -- a real audit target with direct
literature backing: historically underdiagnosed in women (M=163/F=69 in this
data, a ~2.4:1 ratio, less extreme than the classically cited ~4:1 but still
a real skew, not anatomical determinism like prostate_cancer/ovarian_cyst).
Smallest of the 5 conditions (n=232 before class-balancing) but still clears
the pipeline's n>=100 guard.

    python datasets/mimic_prepare.py --data-dir data/raw/mimic   # one-off
    python run.py --dataset mimic_autism --data data/raw/mimic/mimic-raw.csv
"""
import pandas as pd


def load(path='data/raw/mimic/mimic-raw.csv'):
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    df['label'] = df['autism'].map({1: 'autism', 0: 'control'})
    return df


def conditions(df):
    return ['autism']
