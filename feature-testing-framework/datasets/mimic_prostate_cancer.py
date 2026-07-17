"""Loader for MIMIC-IV prostate cancer ICD-coding -- a POSITIVE CONTROL, not
a bias-audit target: prostate cancer is anatomically restricted to males (0
female cases in this data), so the MI test correctly showing gender fully
determines the label here confirms the pipeline detects genuine biological
determinism rather than manufacturing a "shortcut" finding out of it. See
mimic_osteoporosis.py / mimic_autism.py / mimic_cad.py for the real targets.

    python datasets/mimic_prepare.py --data-dir data/raw/mimic   # one-off
    python run.py --dataset mimic_prostate_cancer --data data/raw/mimic/mimic-raw.csv
"""
import pandas as pd


def load(path='data/raw/mimic/mimic-raw.csv'):
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    df['label'] = df['prostate_cancer'].map({1: 'prostate_cancer', 0: 'control'})
    return df


def conditions(df):
    return ['prostate_cancer']
