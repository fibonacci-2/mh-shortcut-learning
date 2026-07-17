#!/usr/bin/env python3
"""One-off: assemble data/raw/umd_crowd/umd-crowd-raw.csv from the UMD
crowd-split tar.gz (umd_reddit_suicidewatch_dataset_v2/crowd/). Only needs to
be run once per fresh extraction of the archive; extract_gender.py and the
umd_crowd loader consume its output, not the raw archive.

    tar xzf umd_reddit_suicidewatch_dataset_v2.tar.gz -C data/raw/umd_crowd
    python datasets/umd_crowd_prepare.py --archive-dir data/raw/umd_crowd/umd_reddit_suicidewatch_dataset_v2

shared_task_posts.csv (train) is ~2M rows / 600MB and includes posts for many
more users than the 993 labeled train users, so it's read in chunks and
filtered down before concatenating -- loading it whole is unnecessary and slow.
"""
import argparse
from pathlib import Path

import pandas as pd

COLS = ['post_id', 'user_id', 'subreddit', 'post_title', 'post_body']


def _load_split(posts_csv, labels_csv, label_col='label', chunksize=None):
    labels = pd.read_csv(labels_csv).rename(columns={'raw_label': label_col})
    users = set(labels['user_id'])

    if chunksize:
        parts = [c[c['user_id'].isin(users)] for c in
                  pd.read_csv(posts_csv, chunksize=chunksize, usecols=COLS)]
        posts = pd.concat(parts, ignore_index=True)
    else:
        posts = pd.read_csv(posts_csv, usecols=COLS)
        posts = posts[posts['user_id'].isin(users)]

    return posts, labels.rename(columns={label_col: 'label'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive-dir', required=True, help='path to the extracted umd_reddit_suicidewatch_dataset_v2/ dir')
    ap.add_argument('--out', default='data/raw/umd_crowd/umd-crowd-raw.csv')
    args = ap.parse_args()

    crowd = Path(args.archive_dir) / 'crowd'
    tr_posts, tr_labels = _load_split(
        crowd / 'train/shared_task_posts.csv', crowd / 'train/crowd_train.csv', chunksize=200_000)
    te_posts, te_labels = _load_split(
        crowd / 'test/shared_task_posts_test.csv', crowd / 'test/crowd_test.csv')

    posts = pd.concat([tr_posts, te_posts], ignore_index=True)
    labels = pd.concat([tr_labels, te_labels], ignore_index=True)

    posts['text'] = (posts['post_title'].fillna('') + '. ' + posts['post_body'].fillna('')).str.strip()
    df = posts.merge(labels, on='user_id', how='left')[['user_id', 'post_id', 'label', 'text']]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} posts, {df['user_id'].nunique():,} users -> {args.out}")
    print(df.drop_duplicates('user_id')['label'].value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()
