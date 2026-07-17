"""Loader for MIMIC-IV coronary artery disease (CAD) ICD-coding -- a real
audit target with direct literature backing: well-documented gender bias in
cardiology diagnosis, where women's "atypical" symptom presentations are
historically under-recognized (M=42,441/F=24,690 in this data). Largest of
the 5 conditions, so the best-powered test of the five.

    python datasets/mimic_prepare.py --data-dir data/raw/mimic   # one-off
    python run.py --dataset mimic_cad --data data/raw/mimic/mimic-raw.csv
"""
import pandas as pd


def load(path='data/raw/mimic/mimic-raw.csv'):
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    df['label'] = df['cad'].map({1: 'cad', 0: 'control'})
    return df


def conditions(df):
    return ['cad']
