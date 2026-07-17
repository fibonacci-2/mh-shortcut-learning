"""Loader for MIMIC-IV osteoporosis ICD-coding -- a real audit target: both
genders present (F=14,454 / M=3,202 admissions overall), skewed but not
anatomically deterministic, unlike prostate_cancer/ovarian_cyst (see those
loaders' docstrings). Text is HPI + Social History, gender-neutralized; see
mimic_prepare.py for how data/raw/mimic/mimic-raw.csv was built.

    python datasets/mimic_prepare.py --data-dir data/raw/mimic   # one-off
    python run.py --dataset mimic_osteoporosis --data data/raw/mimic/mimic-raw.csv
"""
import pandas as pd


def load(path='data/raw/mimic/mimic-raw.csv'):
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    df['label'] = df['osteoporosis'].map({1: 'osteoporosis', 0: 'control'})
    return df


def conditions(df):
    return ['osteoporosis']
