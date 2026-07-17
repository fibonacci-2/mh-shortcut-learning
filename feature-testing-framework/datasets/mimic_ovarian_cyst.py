"""Loader for MIMIC-IV ovarian cyst ICD-coding -- a POSITIVE CONTROL, not a
bias-audit target: ovarian cysts are anatomically restricted to females (3
stray male-coded cases out of 1,295 in this data, likely miscoding), so
gender should legitimately and almost fully determine the label here. See
mimic_osteoporosis.py / mimic_autism.py / mimic_cad.py for the real targets.

    python datasets/mimic_prepare.py --data-dir data/raw/mimic   # one-off
    python run.py --dataset mimic_ovarian_cyst --data data/raw/mimic/mimic-raw.csv
"""
import pandas as pd


def load(path='data/raw/mimic/mimic-raw.csv'):
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    df['label'] = df['ovarian_cyst'].map({1: 'ovarian_cyst', 0: 'control'})
    return df


def conditions(df):
    return ['ovarian_cyst']
