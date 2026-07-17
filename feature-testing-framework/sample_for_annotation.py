#!/usr/bin/env python3
"""Draw a stratified random sample of gender-extraction hits for manual
annotation, to measure precision of each pattern tier in extract_gender.py.
Stratifies by pattern_type first (so rare tiers like pronoun_tag aren't
drowned out by whichever tier fires most) then tops up to the target count.

Usage:
    python sample_for_annotation.py --out data/gender/annotation_sample.csv --n 100
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    'mddl': 'data/gender/mddl-raw.csv',
    'umd': 'data/gender/umd-raw.csv',
    'umd_crowd': 'data/gender/umd-crowd-raw.csv',
    'irf': 'data/gender/irf-raw.csv',
}


def _snippet(text, excerpt, window=120):
    text = str(text)
    i = text.lower().find(str(excerpt).lower())
    if i == -1:
        return text[:2 * window]
    start, end = max(0, i - window), min(len(text), i + len(str(excerpt)) + window)
    return ('…' if start > 0 else '') + text[start:end] + ('…' if end < len(text) else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/gender/annotation_sample.csv')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    frames = []
    for source, path in SOURCES.items():
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        df = df.dropna(subset=['gender']).drop_duplicates(subset=['user_id']).copy()
        df['source'] = source
        frames.append(df[['source', 'user_id', 'gender_source_text', 'matched_excerpt',
                           'gender', 'confidence', 'pattern_type']])
    pool = pd.concat(frames, ignore_index=True)
    pool['snippet'] = pool.apply(lambda r: _snippet(r['gender_source_text'], r['matched_excerpt']), axis=1)

    rng = np.random.RandomState(args.seed)
    per_tier = max(1, args.n // pool['pattern_type'].nunique())
    picked = [grp.sample(min(per_tier, len(grp)), random_state=args.seed)
              for _, grp in pool.groupby('pattern_type')]
    sample = pd.concat(picked)

    remaining = args.n - len(sample)
    if remaining > 0:
        leftover = pool.drop(sample.index)
        sample = pd.concat([sample, leftover.sample(min(remaining, len(leftover)), random_state=args.seed)])

    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle so tiers interleave
    sample = sample.rename(columns={'gender': 'predicted_gender'})
    sample['manual_gender'] = ''  # fill in: male / female / nonbinary / none / unclear
    sample = sample[['source', 'user_id', 'pattern_type', 'confidence', 'predicted_gender',
                      'matched_excerpt', 'snippet', 'manual_gender']]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"{len(sample)} rows -> {args.out}")
    print(sample['pattern_type'].value_counts().to_string())
    print(sample['source'].value_counts().to_string())


if __name__ == '__main__':
    main()
