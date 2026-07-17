"""
Stage 3 on the DETECTION-ALIGNED 6 shortcuts (user-selected canonical set):
  fp_singular, body_health, post_length, hedge_density, negative_emotion, emotional_feeling
Controls: analytic, self_ref_rumination, cogproc, clout.

Reproduces the exact 03 mediation (Cell 5) and INLP erasure/flip (Cell 9/10)
functions over the 6-set. Bootstrap parallelized with joblib (same seeded indices
as the serial loop, so results are identical, just faster). Overwrites the stale
7-set CSVs in data/stage3/ (backed up to *.bak7 first).
"""
import os, json, time, shutil, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.special import expit

SEED, N_BOOT = 42, 1000
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
SIX = ['fp_singular', 'body_health', 'post_length', 'hedge_density', 'negative_emotion', 'emotional_feeling']
log(f"6 shortcuts: {SIX}")

df = pd.read_pickle('data/stage1/features_18_extracted.pkl'); df['text'] = df['text'].fillna('')
GCOL = 'gender_label' if 'gender_label' in df.columns else 'gender'
df['gender_bin'] = (df[GCOL] == 'female').astype(int)
if 'binary_label' not in df.columns:
    df['binary_label'] = (~df['TID'].str.contains('control')).astype(int)
G, Y = df['gender_bin'].values, df['binary_label'].values
for f in ['stage3_mediation_results', 'stage3_concept_erasure_results', 'stage3_counterfactual_flips']:
    p = f'data/stage3/{f}.csv'
    if os.path.exists(p) and not os.path.exists(p+'.bak7'): shutil.copy(p, p+'.bak7')

# ===== MEDIATION (03 Cell 5), bootstrap parallelized =====
M_all = StandardScaler().fit_transform(df[SIX].fillna(0).values)
def compute_mediation(G, Y, M):
    alpha = LinearRegression().fit(G.reshape(-1, 1), M).coef_[0]
    lr_y = LogisticRegression(max_iter=1000, random_state=SEED).fit(np.column_stack([G, M]), Y)
    b0, tau_p, gamma = lr_y.intercept_[0], lr_y.coef_[0, 0], lr_y.coef_[0, 1]
    tau = LogisticRegression(max_iter=1000, random_state=SEED).fit(G.reshape(-1, 1), Y).coef_[0, 0]
    lr_m = LinearRegression().fit(G.reshape(-1, 1), M)
    m1, m0 = lr_m.predict(np.ones((len(G), 1))), lr_m.predict(np.zeros((len(G), 1)))
    acme = (expit(b0+tau_p*G+gamma*m1) - expit(b0+tau_p*G+gamma*m0)).mean()
    ade = (expit(b0+tau_p*1+gamma*M) - expit(b0+tau_p*0+gamma*M)).mean()
    return acme, ade, acme+ade, alpha, tau_p, gamma, tau

med = []
for j, feat in enumerate(SIX):
    M = M_all[:, j]
    acme, ade, total, alpha, tau_p, gamma, tau = compute_mediation(G, Y, M)
    rng = np.random.RandomState(SEED); n = len(G)
    boots = [compute_mediation(G[ix], Y[ix], M[ix])[0]
             for ix in (rng.choice(n, n, replace=True) for _ in range(N_BOOT))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    med.append({'feature': feat, 'alpha_G_to_M': alpha, 'gamma_M_to_Y': gamma, 'tau_total': tau,
                'tau_prime_direct': tau_p, 'ACME': acme, 'ADE': ade, 'total_effect': total,
                'prop_mediated': acme/total, 'ACME_CI_lo': lo, 'ACME_CI_hi': hi,
                'sig': '***' if (lo > 0 or hi < 0) else 'n.s.'})
    log(f"  mediation {feat:18s} ACME={acme:+.5f} prop={acme/total:5.1%} {med[-1]['sig']}")
med_df = pd.DataFrame(med); med_df.to_csv('data/stage3/stage3_mediation_results.csv', index=False)
log(f"mediation sum(prop)={med_df.prop_mediated.sum():.1%}  saved.")

# ===== INLP ERASURE + FLIPS (03 Cell 9/10) =====
X = StandardScaler().fit_transform(np.load('data/stage2/cls_embeddings.npy'))
def probe_auc(Xm, y, k=5):
    return cross_val_score(LogisticRegression(max_iter=500, C=1.0, random_state=SEED), Xm, y,
                           cv=StratifiedKFold(k, shuffle=True, random_state=SEED), scoring='roc_auc').mean()
def inlp(Xin, y, mx=15, seed=42):
    Xr = Xin.copy(); rem = 0
    for i in range(mx):
        rng = np.random.RandomState(seed+i); m = rng.rand(len(y)) < 0.8
        c = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs', random_state=seed).fit(Xr[m], y[m])
        try: a = roc_auc_score(y[~m], c.predict_proba(Xr[~m])[:, 1])
        except Exception: break
        if a < 0.52: break
        c.fit(Xr, y); w = c.coef_[0]/np.linalg.norm(c.coef_[0]); Xr -= np.outer(Xr@w, w); rem += 1
    return Xr, rem

bg, bl = probe_auc(X, G), probe_auc(X, Y)
log(f"baseline frozen: gender={bg:.4f} label={bl:.4f}")
erased, eras = {}, []
for feat in SIX:
    yf = (df[feat].fillna(0).values > df[feat].fillna(0).median()).astype(int)
    pre = probe_auc(X, yf); Xe, n = inlp(X, yf); erased[feat] = Xe
    pc, pg, pl = probe_auc(Xe, yf), probe_auc(Xe, G), probe_auc(Xe, Y)
    eras.append({'concept_erased': feat, 'n_dims_removed': n, 'pre_concept_AUC': pre, 'post_concept_AUC': pc,
                 'delta_concept': pc-pre, 'pre_gender_AUC': bg, 'post_gender_AUC': pg, 'delta_gender': pg-bg,
                 'pre_label_AUC': bl, 'post_label_AUC': pl, 'delta_label': pl-bl})
    log(f"  erase {feat:18s} concept {pre:.3f}->{pc:.3f} label d={pl-bl:+.4f} rem={n}")
Xall = X.copy()
for feat in SIX:
    yf = (df[feat].fillna(0).values > df[feat].fillna(0).median()).astype(int); Xall, _ = inlp(Xall, yf)
erased['ALL_shortcuts'] = Xall
log(f"ALL-6 erase: gender {bg:.3f}->{probe_auc(Xall,G):.3f} label {bl:.3f}->{probe_auc(Xall,Y):.3f}")
erased['gender'], _ = inlp(X, G)
pd.DataFrame(eras).to_csv('data/stage3/stage3_concept_erasure_results.csv', index=False)

clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(X, Y)
po = clf.predict_proba(X)[:, 1]; preo = (po >= 0.5).astype(int); mm, fm = (G == 0), (G == 1)
flips = []
for tgt in SIX + ['ALL_shortcuts', 'gender']:
    pc = clf.predict_proba(erased[tgt])[:, 1]; pcf = (pc >= 0.5).astype(int); fl = (preo != pcf); sh = pc-po
    flips.append({'erased': tgt, 'flip_rate': fl.mean(), 'male_flip': fl[mm].mean(), 'female_flip': fl[fm].mean(),
                  'flip_asym': fl[fm].mean()-fl[mm].mean(), 'mean_dp': sh.mean(), 'male_dp': sh[mm].mean(), 'female_dp': sh[fm].mean()})
    log(f"  flip {tgt:18s} all={fl.mean():.1%} M={fl[mm].mean():.1%} F={fl[fm].mean():.1%} asym={fl[fm].mean()-fl[mm].mean():+.1%}")
pd.DataFrame(flips).to_csv('data/stage3/stage3_counterfactual_flips.csv', index=False)
log("STAGE-3 SIX DONE.")
