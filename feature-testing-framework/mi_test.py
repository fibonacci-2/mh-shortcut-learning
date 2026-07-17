"""Conditional mutual-information test for gender-mediated shortcuts.

For each feature f: compare I(f;Y) to I(f;Y|G). A feature is a gender shortcut
when conditioning on gender G destroys a significant share of its association
with label Y — i.e. its apparent predictive power is mediated by gender rather
than genuine clinical signal. Significance is assessed by permuting gender.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

MI_KW = dict(discrete_features=False, n_neighbors=5)
MIN_DROP_PCT = 50  # a "shortcut" must lose most of its I(f;Y) to conditioning on
# gender, not just a statistically-detectable-at-scale sliver of it -- perm_q
# alone conflates significance with effect size (e.g. a large-n condition can
# make a genuine, gender-independent feature's 7% MI drop "significant" too).


def _mi(X, y, seed):
    """MI per feature column. Zero-variance columns (e.g. a LIWC feature that
    defaults to a constant 0 because the dataset has no LIWC annotation) get
    MI=0 directly -- the kNN-based estimator is degenerate on constant input
    and otherwise returns small spurious nonzero values."""
    mi = np.zeros(X.shape[1])
    live = X.std(axis=0) > 0
    if live.any():
        mi[live] = mutual_info_classif(X[:, live], y, random_state=seed, **MI_KW)
    return mi


def _conditional_mi(X, y, g, seed):
    """I(f;Y|G) = sum_g P(G=g) * I(f;Y | G=g), estimated by stratification.
    Valid by definition of conditional MI (Cover & Thomas, Thm 2.5.2)."""
    mi_cond = np.zeros(X.shape[1])
    for gv in np.unique(g):
        mask = g == gv
        if mask.sum() < 30:  # too few samples for a stable within-stratum estimate
            continue
        mi_cond += mask.mean() * _mi(X[mask], y[mask], seed)
    return mi_cond


def _bh_qvalues(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (q-values) for one family of
    simultaneous tests. With m features tested at once (e.g. 194 Empath
    categories), an uncorrected p<0.05 cutoff lets ~5% of them pass by chance
    alone -- q-values control that false-discovery rate across the batch."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate((p[order] * m / ranks)[::-1])[::-1]
    q = np.empty(m, dtype=float)
    q[order] = np.clip(q_sorted, 0, 1)
    return q


def mi_decomposition(X, y, g, feature_names, n_perm=200, seed=42, fdr=0.05):
    """Run I(f;Y), I(f;G), I(f;Y|G) plus a gender-permutation significance test,
    with Benjamini-Hochberg FDR correction across the feature battery in X.

    X: (n, k) feature matrix. y: binary label array. g: binary gender array.
    Returns one row per feature, sorted by I(f;gender) descending.
    """
    mi_y = _mi(X, y, seed)
    mi_g = _mi(X, g, seed)
    mi_y_given_g = _conditional_mi(X, y, g, seed)
    mi_drop = mi_y - mi_y_given_g

    rng = np.random.RandomState(seed)
    null_drop = np.zeros((n_perm, X.shape[1]))
    for i in range(n_perm):
        g_shuf = rng.permutation(g)
        null_drop[i] = mi_y - _conditional_mi(X, y, g_shuf, i)
    perm_p = (null_drop >= mi_drop).mean(axis=0)
    perm_q = _bh_qvalues(perm_p)

    with np.errstate(divide='ignore', invalid='ignore'):
        drop_pct = np.where(mi_y >= 1e-3, mi_drop / mi_y * 100, np.nan)

    out = pd.DataFrame({
        'feature': feature_names,
        'I(f;Y)': mi_y,
        'I(f;G)': mi_g,
        'I(f;Y|G)': mi_y_given_g,
        'MI_drop': mi_drop,
        'MI_drop_pct': drop_pct,
        'perm_p': perm_p,
        'perm_q': perm_q,
    })
    out['shortcut'] = (out['perm_q'] < fdr) & (out['MI_drop_pct'] >= MIN_DROP_PCT)
    return out.sort_values('I(f;G)', ascending=False).reset_index(drop=True)


def prepare_xyg(df, feature_names, label_col='label', gender_col='gender',
                 positive_label=None, balance_gender=False):
    """Restrict to male/female, class-balance, and build (X, y, g, n) arrays.
    `positive_label` picks the positive class value; omit if label_col is already 0/1.

    balance_gender=True additionally forces each label class to have equal
    male/female counts, so gender and label are independent by construction
    in the resampled data (I(y;g)~=0). Run the MI test on both modes and
    compare: a shortcut that disappears here was an artifact of the label's
    natural gender base-rate skew (fixable by resampling); one that survives
    is a deeper style/gender entanglement that resampling alone won't fix.
    See gender_label_association() for measuring the skew itself."""
    d = df[df[gender_col].isin(['male', 'female'])].copy()
    y = (d[label_col] == positive_label).astype(int) if positive_label is not None else d[label_col].astype(int)
    d = d.assign(_y=y)

    if balance_gender:
        per_class = []
        for v in (0, 1):
            sub = d[d['_y'] == v]
            n_g = sub[gender_col].value_counts().reindex(['male', 'female']).fillna(0).min()
            per_class.append(pd.concat(
                [sub[sub[gender_col] == gv].sample(n=int(n_g), random_state=42) for gv in ('male', 'female')],
                ignore_index=True))
        n = min(len(c) for c in per_class)
        d = pd.concat([c.sample(n=n, random_state=42) for c in per_class], ignore_index=True)
    else:
        n = d['_y'].value_counts().min()
        d = pd.concat([d[d['_y'] == v].sample(n=n, random_state=42) for v in (0, 1)], ignore_index=True)

    X = d[feature_names].fillna(0).to_numpy(dtype=float)
    y = d['_y'].to_numpy()
    g = (d[gender_col] == 'female').to_numpy(dtype=int)
    return X, y, g, len(d)


def gender_label_association(df, label_col='label', gender_col='gender', positive_label=None):
    """2x2 gender x label contingency stats -- the base-rate confound that,
    per the mindset investigation, turned out to be the single best predictor
    of whether a dataset shows gender-mediated shortcuts at all. Report this
    up front instead of discovering it by hand after the fact.

    odds_ratio > 1 means the positive label skews male, < 1 skews female.
    cramers_v (== |phi| for a 2x2 table) is 0 when gender and label are
    independent, 1 when perfectly confounded.
    """
    d = df[df[gender_col].isin(['male', 'female'])].copy()
    y = (d[label_col] == positive_label).astype(int) if positive_label is not None else d[label_col].astype(int)
    ct = pd.crosstab(d[gender_col], y).reindex(index=['male', 'female'], columns=[0, 1], fill_value=0)

    a, b = ct.loc['male', 1], ct.loc['male', 0]
    c, e = ct.loc['female', 1], ct.loc['female', 0]
    odds_ratio = (a * e) / (b * c) if b * c > 0 else np.nan

    n = ct.values.sum()
    if n > 0 and ct.values.min() >= 0:
        chi2, chi2_p, _, _ = chi2_contingency(ct, correction=False)
        cramers_v = np.sqrt(chi2 / n)
    else:
        chi2_p, cramers_v = np.nan, np.nan

    return {
        'odds_ratio_male_pos': odds_ratio,
        'cramers_v': cramers_v,
        'chi2_p': chi2_p,
        'n_male': int(a + b),
        'n_female': int(c + e),
    }
