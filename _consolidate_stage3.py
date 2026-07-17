"""
Consolidate Stage 3 + frozen-DistilRoBERTa probing onto the CANONICAL 8 shortcuts
(stage1_results.json shortcut_features). The on-disk CSVs are stale: they were
produced when shortcut_features was a different 7-set (incl. excl_intensifiers,
social_relational; excl. analytic, body_health, self_ref_rumination). The notebook
*code* already targets the canonical 8 via SHORTCUT_FEATURES, so we re-run the
identical mediation (03 Cell 5) and INLP erasure/flip (03 Cell 9/10, recovered
from git f5f2aa5^) functions verbatim over the current set.

Overwrites: data/stage3/{stage3_mediation_results,stage3_concept_erasure_results,
stage3_counterfactual_flips}.csv  and writes data/stage2/frozen_probe_8.csv.
"""
import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.special import expit

SEED, N_BOOT = 42, 1000
np.random.seed(SEED)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

s1 = json.load(open('data/stage1/stage1_results.json'))
SHORTCUTS, CONTROLS = s1['shortcut_features'], s1['passed_controls']
log(f"canonical shortcuts ({len(SHORTCUTS)}): {SHORTCUTS}")

df = pd.read_pickle('data/stage1/features_18_extracted.pkl'); df['text'] = df['text'].fillna('')
GCOL = 'gender_label' if 'gender_label' in df.columns else 'gender'
df['gender_bin'] = (df[GCOL] == 'female').astype(int)
if 'binary_label' not in df.columns:
    df['binary_label'] = (~df['TID'].str.contains('control')).astype(int)
G = df['gender_bin'].values; Y = df['binary_label'].values

# ============================ MEDIATION (03 Cell 5) ==========================
M_all = StandardScaler().fit_transform(df[SHORTCUTS].fillna(0).values)

def compute_mediation(G, Y, M):
    lr_m = LinearRegression().fit(G.reshape(-1, 1), M); alpha = lr_m.coef_[0]
    lr_y = LogisticRegression(max_iter=1000, random_state=SEED).fit(np.column_stack([G, M]), Y)
    b0, tau_p, gamma = lr_y.intercept_[0], lr_y.coef_[0, 0], lr_y.coef_[0, 1]
    tau = LogisticRegression(max_iter=1000, random_state=SEED).fit(G.reshape(-1, 1), Y).coef_[0, 0]
    m1, m0 = lr_m.predict(np.ones((len(G), 1))), lr_m.predict(np.zeros((len(G), 1)))
    acme = (expit(b0+tau_p*G+gamma*m1) - expit(b0+tau_p*G+gamma*m0)).mean()
    ade  = (expit(b0+tau_p*1+gamma*M) - expit(b0+tau_p*0+gamma*M)).mean()
    return acme, ade, acme+ade, alpha, tau_p, gamma, tau

med = []
for j, feat in enumerate(SHORTCUTS):
    M = M_all[:, j]
    acme, ade, total, alpha, tau_p, gamma, tau = compute_mediation(G, Y, M)
    rng = np.random.RandomState(SEED); n = len(G); boots = []
    for _ in range(N_BOOT):
        idx = rng.choice(n, n, replace=True)
        boots.append(compute_mediation(G[idx], Y[idx], M[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    med.append({'feature': feat, 'alpha_G_to_M': alpha, 'gamma_M_to_Y': gamma,
                'tau_total': tau, 'tau_prime_direct': tau_p, 'ACME': acme, 'ADE': ade,
                'total_effect': total, 'prop_mediated': acme/total if abs(total) > 1e-10 else 0,
                'ACME_CI_lo': lo, 'ACME_CI_hi': hi, 'sig': '***' if (lo > 0 or hi < 0) else 'n.s.'})
    log(f"  mediation {feat:20s} ACME={acme:+.5f} prop={acme/total:.1%} {med[-1]['sig']}")
med_df = pd.DataFrame(med); med_df.to_csv('data/stage3/stage3_mediation_results.csv', index=False)
log(f"mediation total prop (sum) = {med_df['prop_mediated'].sum():.1%}; saved.")

# ====================== INLP ERASURE + FLIPS (03 Cell 9/10) ==================
X_scaled = StandardScaler().fit_transform(np.load('data/stage2/cls_embeddings.npy'))

def probe_auc(X, y, n_splits=5):
    return cross_val_score(LogisticRegression(max_iter=500, C=1.0, random_state=SEED), X, y,
                           cv=StratifiedKFold(n_splits, shuffle=True, random_state=SEED), scoring='roc_auc').mean()

def inlp_erase(X, y, max_iters=15, seed=42):
    Xr = X.copy(); removed = 0
    for i in range(max_iters):
        rng = np.random.RandomState(seed+i); mask = rng.rand(len(y)) < 0.8
        clf = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs', random_state=seed).fit(Xr[mask], y[mask])
        try: auc = roc_auc_score(y[~mask], clf.predict_proba(Xr[~mask])[:, 1])
        except Exception: break
        if auc < 0.52: break
        clf.fit(Xr, y); w = clf.coef_[0]/np.linalg.norm(clf.coef_[0])
        Xr -= np.outer(Xr @ w, w); removed += 1
    return Xr, removed

base_gender_auc, base_label_auc = probe_auc(X_scaled, G), probe_auc(X_scaled, Y)
log(f"baseline frozen probes: gender={base_gender_auc:.4f} label={base_label_auc:.4f}")

erased, eras = {}, []
for feat in SHORTCUTS:
    yf = (df[feat].fillna(0).values > df[feat].fillna(0).median()).astype(int)
    pre = probe_auc(X_scaled, yf)
    Xe, n = inlp_erase(X_scaled, yf); erased[feat] = Xe
    pc, pg, pl = probe_auc(Xe, yf), probe_auc(Xe, G), probe_auc(Xe, Y)
    eras.append({'concept_erased': feat, 'n_dims_removed': n,
                 'pre_concept_AUC': pre, 'post_concept_AUC': pc, 'delta_concept': pc-pre,
                 'pre_gender_AUC': base_gender_auc, 'post_gender_AUC': pg, 'delta_gender': pg-base_gender_auc,
                 'pre_label_AUC': base_label_auc, 'post_label_AUC': pl, 'delta_label': pl-base_label_auc})
    log(f"  erase {feat:20s} concept {pre:.3f}->{pc:.3f} | label d={pl-base_label_auc:+.4f} | removed {n}")

X_all = X_scaled.copy()
for feat in SHORTCUTS:
    yf = (df[feat].fillna(0).values > df[feat].fillna(0).median()).astype(int)
    X_all, _ = inlp_erase(X_all, yf)
erased['ALL_shortcuts'] = X_all
log(f"ALL-erase: gender {base_gender_auc:.3f}->{probe_auc(X_all,G):.3f} label {base_label_auc:.3f}->{probe_auc(X_all,Y):.3f}")
erased['gender'], _ = inlp_erase(X_scaled, G)
pd.DataFrame(eras).to_csv('data/stage3/stage3_concept_erasure_results.csv', index=False)

# flips
clf_orig = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(X_scaled, Y)
probs_orig = clf_orig.predict_proba(X_scaled)[:, 1]; preds_orig = (probs_orig >= 0.5).astype(int)
mm, fm = (G == 0), (G == 1)
flips = []
for tgt in SHORTCUTS + ['ALL_shortcuts', 'gender']:
    pc = clf_orig.predict_proba(erased[tgt])[:, 1]; pcf = (pc >= 0.5).astype(int)
    fl = (preds_orig != pcf); sh = pc - probs_orig
    flips.append({'erased': tgt, 'flip_rate': fl.mean(), 'male_flip': fl[mm].mean(),
                  'female_flip': fl[fm].mean(), 'flip_asym': fl[fm].mean()-fl[mm].mean(),
                  'mean_dp': sh.mean(), 'male_dp': sh[mm].mean(), 'female_dp': sh[fm].mean()})
    log(f"  flip {tgt:20s} all={fl.mean():.1%} M={fl[mm].mean():.1%} F={fl[fm].mean():.1%} asym={fl[fm].mean()-fl[mm].mean():+.1%}")
pd.DataFrame(flips).to_csv('data/stage3/stage3_counterfactual_flips.csv', index=False)

# ================= FROZEN DISTILROBERTA PROBING (fill --- cells) =============
rows = []
for f in SHORTCUTS + CONTROLS:
    yf = (df[f].fillna(0).values > df[f].fillna(0).median()).astype(int)
    rows.append({'target': f, 'role': 'shortcut' if f in SHORTCUTS else 'control', 'frozen_auc': probe_auc(X_scaled, yf)})
rows.append({'target': 'gender', 'role': 'gender', 'frozen_auc': base_gender_auc})
rows.append({'target': 'binary_label', 'role': 'label', 'frozen_auc': base_label_auc})
fp = pd.DataFrame(rows); fp.to_csv('data/stage2/frozen_probe_8.csv', index=False)
log("frozen DistilRoBERTa probe (fills --- cells):")
for r in rows: log(f"    {r['target']:20s} {r['frozen_auc']:.4f}")
log("CONSOLIDATION DONE.")
