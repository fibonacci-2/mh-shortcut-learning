#!/usr/bin/env python3
"""One-off: assemble data/raw/mddl/mddl-raw.csv from the MDDL (Multimodal
Depressive Dictionary Learning, Shen et al. 2017) Twitter dataset's `labeled`
split -- only needs to be run once per fresh extraction of the archive (a
PPMd-compressed zip; extract with `7zz x Dataset_MDDL.zip "Dataset/labeled/*"`,
plain `unzip`/Python's zipfile don't support PPMd).

Each user has up to three text sources, joined by Twitter screen_name:
  - data/users/{screen_name}.json      -- profile object, bio in 'description'
  - data/tweet/{tweet_id}.json         -- the single seed/self-disclosure tweet
                                           that got the user flagged (nested
                                           .user.screen_name links it to a user)
  - data/timeline/{screen_name}created_at{ts}.json -- JSON-lines tweet history;
                                           a user can have multiple such files

We concatenate whichever of these exist per user (bio + seed tweet + full
timeline) to maximize self-disclosure recall, the same way umd_crowd_prepare.py
aggregates a user's full post history rather than a single excerpt.

    python datasets/mddl_prepare.py --archive-dir data/raw/mddl/Dataset
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_TIMELINE_RE = re.compile(r'^(.*?)created_at(-?\d+)\.json$')


def _read_jsonl_texts(path):
    texts = []
    for line in path.read_text(errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            texts.append(json.loads(line).get('text', ''))
        except json.JSONDecodeError:
            continue
    return texts


def _load_class(class_dir, label):
    class_dir = Path(class_dir)
    texts = defaultdict(list)  # screen_name -> list of text snippets

    for f in tqdm(list((class_dir / 'data/users').glob('*.json')), desc=f'{label} bios'):
        try:
            bio = json.loads(f.read_text(errors='ignore')).get('description')
        except json.JSONDecodeError:
            bio = None
        if bio:
            texts[f.stem].append(bio)

    for f in tqdm(list((class_dir / 'data/tweet').glob('*.json')), desc=f'{label} seed tweets'):
        try:
            d = json.loads(f.read_text(errors='ignore').splitlines()[0])
        except (json.JSONDecodeError, IndexError):
            continue
        sn = d.get('user', {}).get('screen_name')
        if sn and d.get('text'):
            texts[sn].append(d['text'])

    for f in tqdm(list((class_dir / 'data/timeline').glob('*.json')), desc=f'{label} timelines'):
        m = _TIMELINE_RE.match(f.name)
        if not m:
            continue
        texts[m.group(1)].extend(_read_jsonl_texts(f))

    rows = [{'user_id': sn, 'label': label, 'text': '. '.join(snippets)}
            for sn, snippets in texts.items()]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive-dir', required=True, help='path to the extracted Dataset/ dir (containing labeled/)')
    ap.add_argument('--out', default='data/raw/mddl/mddl-raw.csv')
    args = ap.parse_args()

    labeled = Path(args.archive_dir) / 'labeled'
    pos = _load_class(labeled / 'positive', 'depression')
    neg = _load_class(labeled / 'negative', 'control')
    df = pd.concat([pos, neg], ignore_index=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} users -> {args.out}")
    print(df['label'].value_counts().to_string())
    print(df['text'].str.len().describe())


if __name__ == '__main__':
    main()
