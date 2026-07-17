#!/usr/bin/env python3
"""CLI: extract features + run the gender-shortcut MI test on a dataset.

Usage:
    python run.py --dataset synthetic
    python run.py --dataset mindset --data data/strong-ann.csv
    python run.py --dataset mindset --data data/strong-ann.csv --condition depression --out results.csv
    python run.py --dataset umd_demographics --feature-set empath
    python run.py --dataset umd_demographics --feature-set both --out results.csv
"""
import argparse
import numpy as np
import pandas as pd

from features import compute_features, FEATURE_NAMES, REFERENCE_FEATURES
from empath_features import compute_empath_features, EMPATH_CATEGORIES
from mi_test import mi_decomposition, prepare_xyg, gender_label_association
from datasets import REGISTRY

# name -> (function to add feature columns, list of feature column names)
FEATURE_SETS = {
    'core':   (compute_features, FEATURE_NAMES),
    'empath': (compute_empath_features, EMPATH_CATEGORIES),
    'both':   (lambda df: compute_empath_features(compute_features(df)),
               FEATURE_NAMES + EMPATH_CATEGORIES),
}


def run_condition(df, condition, feature_names, n_perm, fdr, label_col='label', gender_col='gender'):
    d = df[df[label_col].isin([condition, 'control'])]

    assoc = gender_label_association(d, label_col=label_col, gender_col=gender_col, positive_label=condition)

    X, y, g, n = prepare_xyg(d, feature_names, label_col=label_col,
                              gender_col=gender_col, positive_label=condition)
    if n < 100:
        return None
    res = mi_decomposition(X, y, g, feature_names, n_perm=n_perm, fdr=fdr)
    res['condition'] = condition
    res['n'] = n
    res['is_reference_feature'] = res['feature'].isin(REFERENCE_FEATURES)
    for k, v in assoc.items():
        res[k] = v

    # gender-balanced-within-label re-test: isolates whether a flagged shortcut
    # is a base-rate artifact (would evaporate with different gender base rates
    # at deployment) or a genuine style/gender entanglement (survives even when
    # gender no longer predicts the label at all). See prepare_xyg's docstring.
    Xb, yb, gb, nb = prepare_xyg(d, feature_names, label_col=label_col, gender_col=gender_col,
                                  positive_label=condition, balance_gender=True)
    res['n_balanced'] = nb
    if nb >= 100:
        res_b = mi_decomposition(Xb, yb, gb, feature_names, n_perm=n_perm, fdr=fdr)
        res_b = res_b.set_index('feature')[['MI_drop_pct', 'perm_q', 'shortcut']]
        res_b.columns = ['MI_drop_pct_balanced', 'perm_q_balanced', 'shortcut_balanced']
        res = res.merge(res_b, left_on='feature', right_index=True, how='left')
    else:
        res['MI_drop_pct_balanced'] = np.nan
        res['perm_q_balanced'] = np.nan
        res['shortcut_balanced'] = np.nan

    def classify(row):
        if not row['shortcut']:
            return 'none'
        if pd.isna(row['shortcut_balanced']):
            return 'undetermined (balanced n too small)'
        return 'genuine' if row['shortcut_balanced'] else 'base_rate_artifact'
    res['shortcut_type'] = res.apply(classify, axis=1)

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=REGISTRY)
    ap.add_argument('--data', default=None, help='path to the dataset file')
    ap.add_argument('--condition', help='single condition to test; default = all found')
    ap.add_argument('--feature-set', choices=FEATURE_SETS, default='core',
                     help='core = 18 curated features; empath = 194 Empath categories '
                          '(no LIWC needed); both = union')
    ap.add_argument('--n-perm', type=int, default=200)
    ap.add_argument('--fdr', type=float, default=0.05, help='Benjamini-Hochberg FDR threshold across the feature battery')
    ap.add_argument('--out', default=None, help='CSV path to save results')
    args = ap.parse_args()

    loader = REGISTRY[args.dataset]
    df = loader.load(args.data) if args.data else loader.load()
    add_features, feature_names = FEATURE_SETS[args.feature_set]
    df = add_features(df)

    targets = [args.condition] if args.condition else loader.conditions(df)
    print(f"Feature set: {args.feature_set} ({len(feature_names)} features)")
    print(f"Running MI test on {len(targets)} condition(s): {targets}")

    results = []
    for cond in targets:
        print(f"  {cond} ...", end=' ', flush=True)
        res = run_condition(df, cond, feature_names, args.n_perm, args.fdr)
        if res is None:
            print('skipped (too few samples)')
            continue
        cv = res['cramers_v'].iloc[0]
        orat = res['odds_ratio_male_pos'].iloc[0]
        print(f"n={res['n'].iloc[0]} (gender-balanced n={res['n_balanced'].iloc[0]}), "
              f"gender~label Cramer's V={cv:.3f} (OR male-positive={orat:.2f}), "
              f"shortcuts={res['shortcut'].sum()}")
        results.append(res)

    if not results:
        print("No condition had enough samples.")
        return

    out = pd.concat(results, ignore_index=True)
    pd.set_option('display.width', 160)
    flagged_cols = ['dataset', 'condition', 'feature', 'n', 'MI_drop_pct', 'MI_drop_pct_balanced', 'shortcut_type']
    flagged_cols = [c for c in flagged_cols if c in out.columns]
    print("\n" + out[out['shortcut']][flagged_cols].to_string(index=False) if out['shortcut'].any()
          else "\nNo shortcuts flagged.")

    summary = (out.groupby('feature')['shortcut'].sum()
               .rename('n_conditions_flagged').sort_values(ascending=False))
    summary = summary[summary > 0]
    print("\nShortcut frequency across conditions:")
    print(summary.to_string() if len(summary) else "(none)")

    print("\nShortcut type breakdown (skewed-mode flags, reclassified by the gender-balanced re-test):")
    type_summary = out[out['shortcut']]['shortcut_type'].value_counts()
    print(type_summary.to_string() if len(type_summary) else "(none)")

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nSaved full results ({len(out)} rows) to {args.out}")


if __name__ == '__main__':
    main()
