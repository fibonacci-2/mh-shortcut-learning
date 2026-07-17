#!/usr/bin/env python3
"""Build one combined CSV of every row still missing real LIWC-22 annotation,
for a single batch run through the LIWC-22 CLI.

mindset.py (data/strong-ann.csv) and umd_demographics.py (data/raw/umd_demographics/
umd_standardized_demographics.csv) already ship real LIWC-22 columns from a
prior external run -- skipped here. Everything else (umd, umd_crowd, irf_belong/
irf_burden share one source file, and the 5 mimic_* conditions) has never been
run through LIWC, so features.py's LIWC-derived columns silently default to 0
for them.

Each row gets a stable (source_file, source_row_index) pair so LIWC-22's
output can be split back and merged onto the original per-dataset raw CSVs by
row position afterward -- source_row_index is 0-based matching the row order
each dataset loader's pd.read_csv() sees.

    python prepare_liwc_batch.py --out data/raw/liwc_batch.csv
"""
import argparse
import csv

SOURCES = [
    ('umd', 'data/gender/umd-raw.csv'),
    ('umd_crowd', 'data/gender/umd-crowd-raw.csv'),
    ('irf', 'data/gender/irf-raw.csv'),
    ('mimic', 'data/raw/mimic/mimic-raw.csv'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/raw/liwc_batch.csv')
    args = ap.parse_args()

    csv.field_size_limit(10_000_000)
    n_written = 0
    with open(args.out, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['liwc_row_id', 'source_file', 'source_row_index', 'text'])

        for tag, path in SOURCES:
            with open(path, newline='', encoding='utf-8') as in_f:
                reader = csv.DictReader(in_f)
                n_source = 0
                for i, row in enumerate(reader):
                    text = (row.get('text') or '').replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                    writer.writerow([n_written, tag, i, text])
                    n_written += 1
                    n_source += 1
            print(f'{tag:12s} {path:40s} {n_source:>8,d} rows')

    print(f'\n{n_written:,} total rows -> {args.out}')


if __name__ == '__main__':
    main()
