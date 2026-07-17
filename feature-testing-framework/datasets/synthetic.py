"""Synthetic dataset for smoke-testing the pipeline without real data.

Mirrors the real confound structure: label has a differential base rate by
gender (e.g. condition skews female), and first-person-singular rate depends
on gender ONLY, not label. So fp_singular looks predictive of label when
pooled, purely because both correlate with gender independently -- a textbook
shortcut, which conditioning on gender should erase. emo_neg depends on label
only (gender-independent) and should survive conditioning as genuine signal.
"""
import numpy as np
import pandas as pd


def load(path=None, n=1500, seed=0):
    rng = np.random.RandomState(seed)
    gender = rng.choice(['male', 'female'], size=n)
    is_female = (gender == 'female').astype(int)
    label = rng.binomial(1, 0.30 + 0.40 * is_female)  # differential base rate by gender

    emo_neg = label * 3 + rng.normal(0, 1, n)  # genuine signal: depends on label only
    fp_rate = 0.1 + 0.3 * is_female  # shortcut: depends on gender only

    texts = []
    for i in range(n):
        n_fp = round(30 * fp_rate[i])
        words = ['i'] * n_fp + ['x'] * (30 - n_fp)
        rng.shuffle(words)
        texts.append(' '.join(words))

    return pd.DataFrame({
        'text': texts,
        'label': np.where(label == 1, 'condition', 'control'),
        'gender': gender,
        'emo_neg': emo_neg,
    })


def conditions(df):
    return ['condition']
