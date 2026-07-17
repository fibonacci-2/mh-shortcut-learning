"""Empath (Fast, Chen & Bernstein 2016) lexical-category features -- an
open-source, text-only analogue of LIWC: 194 pre-built categories, computed
directly from raw text with no external annotation step. Unlike the
LIWC-dependent features in features.py (which default to 0 for any dataset
nobody has run real LIWC on), this gives every dataset full feature coverage.
"""
import pandas as pd
from empath import Empath

_LEXICON = Empath()
EMPATH_CATEGORIES = sorted(_LEXICON.cats)


def compute_empath_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all 194 Empath category scores (fraction of tokens matching each
    category) as new columns. Requires a 'text' column."""
    scores = df['text'].apply(lambda t: _LEXICON.analyze(str(t), normalize=True) or {})
    scores_df = pd.DataFrame(list(scores), columns=EMPATH_CATEGORIES, index=df.index).fillna(0.0)
    df = df.drop(columns=[c for c in EMPATH_CATEGORIES if c in df.columns])
    return pd.concat([df, scores_df], axis=1)
