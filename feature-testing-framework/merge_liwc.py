#!/usr/bin/env python3
"""Split the LIWC-22 CLI's batch output (prepare_liwc_batch.py's input, run
through LIWC-22 externally) back onto each dataset's raw CSV, keyed by the
(source_file, source_row_index) pair prepare_liwc_batch.py stamped on every
row -- not by row position, since LIWC-22 silently drops rows with no
dictionary-matchable tokens (emoticon-only / punctuation-only text, ~0.3% of
rows here) rather than emitting a zero row for them. Those get filled with
WC=0 and every category=0, which is the correct value anyway.

Overwrites each raw CSV in place with the LIWC-22 columns appended -- no
dataset loader needs to change, since features.py's LIWC lookups (features.py
:85's `get(col)`) just start finding real columns instead of defaulting to 0.

    python merge_liwc.py --annotated data/raw/liwc_batch_annotated.csv
"""
import argparse
import csv
from pathlib import Path

csv.field_size_limit(10_000_000)

SOURCES = [
    ('umd', 'data/gender/umd-raw.csv'),
    ('umd_crowd', 'data/gender/umd-crowd-raw.csv'),
    ('irf', 'data/gender/irf-raw.csv'),
    ('mimic', 'data/raw/mimic/mimic-raw.csv'),
]

# columns carried in from prepare_liwc_batch.py that aren't LIWC output
_NON_LIWC_COLS = {'liwc_row_id', 'source_file', 'source_row_index', 'text'}


def load_annotated(path):
    """Returns {source_file: {source_row_index: {liwc_col: value}}} and the
    ordered list of LIWC column names."""
    by_source = {tag: {} for tag, _ in SOURCES}
    liwc_cols = None
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        liwc_cols = [c for c in reader.fieldnames if c not in _NON_LIWC_COLS]
        for row in reader:
            sf = row['source_file']
            sri = int(row['source_row_index'])
            by_source[sf][sri] = {c: row[c] for c in liwc_cols}
    return by_source, liwc_cols


def merge_one(path, rows_by_index, liwc_cols):
    path = Path(path)
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    zero_row = {c: ('0' if c != 'Segment' else '') for c in liwc_cols}
    new_header = header + liwc_cols
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    n_filled, n_zero = 0, 0
    with open(tmp_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(new_header)
        for i, row in enumerate(rows):
            liwc_row = rows_by_index.get(i)
            if liwc_row is None:
                liwc_row = zero_row
                n_zero += 1
            else:
                n_filled += 1
            writer.writerow(row + [liwc_row[c] for c in liwc_cols])
    tmp_path.replace(path)
    return len(rows), n_filled, n_zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotated', default='data/raw/liwc_batch_annotated.csv')
    args = ap.parse_args()

    print(f'Loading {args.annotated} ...')
    by_source, liwc_cols = load_annotated(args.annotated)
    print(f'{len(liwc_cols)} LIWC columns: {liwc_cols}\n')

    for tag, path in SOURCES:
        n_total, n_filled, n_zero = merge_one(path, by_source[tag], liwc_cols)
        print(f'{tag:12s} {path:35s} rows={n_total:>7,d}  liwc_filled={n_filled:>7,d}  zero_filled={n_zero:>4,d}')


if __name__ == '__main__':
    main()
