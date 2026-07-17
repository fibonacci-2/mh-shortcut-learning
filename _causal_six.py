"""Re-run the Stage-8 causal fairness intervention with the SIX canonical shortcuts
(not the 8 from stage1_results.json), for consistency with the 6-feature paper.
LEACE-erase the 6-feature subspace from the fine-tuned [CLS], run the unchanged
head, re-measure equalized odds. Conditions: baseline, shortcut-erase(6),
gender-erase, random-erase(rank-6, 10 seeds). Overwrites data/fairness/causal_intervention.csv
and regenerates the figure."""
import os, warnings
warnings.filterwarnings('ignore'); os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from transformers import AutoModelForSequenceClassification

OUT = 'data/fairness'
SIX = ['fp_singular', 'body_health', 'post_length', 'hedge_density', 'negative_emotion', 'emotional_feeling']
MODELS = {'distilroberta-ft': ('data/stage2_finetuned/ft_model', 'data/stage2_finetuned/cls_embeddings_finetuned.npy'),
          'mentalroberta-ft': ('data/stage2_mentalroberta/ft_model', 'data/stage2_mentalroberta/finetuned/cls_embeddings.npy')}

df = pd.read_pickle('data/stage1/features_18_extracted.pkl'); df['text'] = df['text'].fillna('')
GCOL = 'gender_label' if 'gender_label' in df.columns else 'gender'
if 'binary_label' not in df.columns: df['binary_label'] = (~df['TID'].str.contains('control')).astype(int)
y_all = df['binary_label'].values.astype(int); g_all = (df[GCOL].values == 'female').astype(int)
Zs = StandardScaler().fit_transform(df[SIX].fillna(0).values.astype(float))
_, va = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y_all)
tr = np.setdiff1d(np.arange(len(df)), va)

def leace_fit(X, Z, eps=1e-3):
    X = torch.as_tensor(X, dtype=torch.float64); Z = torch.as_tensor(Z, dtype=torch.float64)
    if Z.ndim == 1: Z = Z[:, None]
    mu = X.mean(0); Xc = X - mu; Zc = Z - Z.mean(0); n = X.shape[0]
    S = (Xc.T @ Xc)/n; S = S + eps*torch.trace(S)/S.shape[0]*torch.eye(S.shape[0], dtype=torch.float64)
    lam, V = torch.linalg.eigh(S); lam = torch.clamp(lam, min=1e-12)
    half = V@torch.diag(lam.sqrt())@V.T; ih = V@torch.diag(lam.rsqrt())@V.T
    U, Sg, _ = torch.linalg.svd(ih @ ((Xc.T@Zc)/n), full_matrices=False)
    r = int((Sg > 1e-6*Sg.max()).sum()) if Sg.numel() else 0
    Q = U[:, :r]; P = torch.eye(S.shape[0], dtype=torch.float64) - Q@Q.T
    return mu, half@P@ih, r

def leace_apply(X, fit):
    mu, A, _ = fit; X = torch.as_tensor(X, dtype=torch.float64); return ((X-mu)@A.T+mu).numpy()

def eo(prob, yt, g, thr=0.5):
    yp = (prob >= thr).astype(int)
    def ft(m):
        tn, fp, fn, tp = confusion_matrix(yt[m], yp[m], labels=[0, 1]).ravel()
        return fp/(fp+tn), tp/(tp+fn)
    ff, tf = ft(g == 1); fm, tm = ft(g == 0)
    return roc_auc_score(yt, prob), max(abs(ff-fm), abs(tf-tm))

rows = []
for name, (mdir, clsp) in MODELS.items():
    model = AutoModelForSequenceClassification.from_pretrained(mdir, dtype=torch.float32).eval()
    head = model.classifier
    @torch.no_grad()
    def hp(cls):
        x = torch.as_tensor(cls, dtype=torch.float32)
        x = torch.tanh(head.dense(head.dropout(x)))
        return torch.softmax(head.out_proj(head.dropout(x)), -1)[:, 1].numpy()
    cls = np.load(clsp).astype(np.float32); ct, cv = cls[tr], cls[va]
    yv, gv = y_all[va], g_all[va]
    a0, e0 = eo(hp(cv), yv, gv); rows.append([name, 'baseline', 0, e0, a0])
    fs = leace_fit(ct, Zs[tr]); a, e = eo(hp(leace_apply(cv, fs)), yv, gv); rows.append([name, 'shortcut-erase', fs[2], e, a])
    fg = leace_fit(ct, g_all[tr].astype(float)); a, e = eo(hp(leace_apply(cv, fg)), yv, gv); rows.append([name, 'gender-erase', fg[2], e, a])
    es, as_ = [], []
    for s in range(10):
        Zr = np.random.default_rng(s).standard_normal((len(df), len(SIX)))
        a, e = eo(hp(leace_apply(cv, leace_fit(ct, Zr[tr]))), yv, gv); es.append(e); as_.append(a)
    rows.append([name, 'random-erase', len(SIX), np.mean(es), np.mean(as_), np.std(es)])
    print(f"{name}: base EO={e0:.3f} | shortcut={rows[-3][3]:.3f} | gender={rows[-2][3]:.3f} | random={np.mean(es):.3f}+-{np.std(es):.3f}", flush=True)
    del model

res = pd.DataFrame([r+[np.nan] if len(r) == 5 else r for r in rows],
                   columns=['model', 'condition', 'rank', 'eo', 'auc', 'eo_sd'])
res.to_csv(f'{OUT}/causal_intervention.csv', index=False)
print(res.to_string(index=False))

# figure
order = ['baseline', 'shortcut-erase', 'random-erase', 'gender-erase']
col = {'baseline': '#555', 'shortcut-erase': '#c44e52', 'random-erase': '#b0b0b0', 'gender-erase': '#4c72b0'}
models = list(MODELS); fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); w = .2; x = np.arange(len(models))
for j, c in enumerate(order):
    vals = [res[(res.model == m) & (res.condition == c)].eo.values[0] for m in models]
    err = [res[(res.model == m) & (res.condition == c)].eo_sd.values[0] if c == 'random-erase' else 0 for m in models]
    err = [0 if pd.isna(e) else e for e in err]
    ax[0].bar(x+(j-1.5)*w, vals, w, yerr=err, capsize=3, label=c, color=col[c])
ax[0].set_xticks(x); ax[0].set_xticklabels(models); ax[0].set_ylabel('Equalized-odds violation')
ax[0].set_title('(a) Erasing the shortcut subspace halves the gap'); ax[0].legend(fontsize=8.5)
d = res[res.model == 'distilroberta-ft']
for c in order:
    r = d[d.condition == c].iloc[0]; ax[1].scatter(r.eo, r.auc, s=160, color=col[c], edgecolor='k', zorder=3)
    ax[1].annotate(c, (r.eo, r.auc), textcoords='offset points', xytext=(8, 6), fontsize=9)
ax[1].set_xlabel('Equalized-odds violation  (<- fairer)'); ax[1].set_ylabel('Overall label AUC')
ax[1].set_title('(b) Accuracy-fairness frontier (distilroberta-ft)'); ax[1].invert_xaxis(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/causal_intervention.png', dpi=150, bbox_inches='tight')
print('saved figure')
