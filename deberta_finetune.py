"""
DeBERTa-v3 fine-tune + probing — standalone, multi-GPU, robust to transformers v5.

Closes the only pending cell of the Stage-2 sweep (the 02d FT half). Uses a plain
PyTorch loop with nn.DataParallel across all visible GPUs, sidestepping the v5
Trainer callback regression (`on_train_begin ... before on_evaluate`) and the
`group_by_length` rename. DeBERTa-v3 loads as fp32 (its config defaults to fp16,
which NaNs under AdamW); speed comes from fp16 AMP autocast over fp32 master
weights + 4-GPU data parallelism.

Outputs (data/stage2_deberta/finetuned/):
  ft_model/                      saved fine-tuned classifier
  cls_embeddings.npy             FT [CLS] for all 32,200 posts (last_hidden[:,0])
  probing_results.csv            final-layer 5-fold logistic probe AUC
  layerwise_probing.csv          depth-resolved probe AUC (13 points)
Run:  CUDA_VISIBLE_DEVICES=0,1,2,3 ./.venv/bin/python deberta_finetune.py
"""
import os, json, time, warnings
warnings.filterwarnings('ignore'); os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModel, get_linear_schedule_with_warmup)

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = 'cuda'
MODEL_NAME = 'microsoft/deberta-v3-base'
MAX_LEN, BATCH, EPOCHS, LR = 256, 64, 3, 2e-5
OUT = 'data/stage2_deberta/finetuned'; os.makedirs(OUT, exist_ok=True)
FT_DIR = f'{OUT}/ft_model'; os.makedirs(FT_DIR, exist_ok=True)
NGPU = torch.cuda.device_count()
log(f"GPUs={NGPU}  effective batch={BATCH}")

# ── data + identical split ───────────────────────────────────────────────────
s1 = json.load(open('data/stage1/stage1_results.json'))
SHORTCUTS, CONTROLS = s1['shortcut_features'], s1['passed_controls']
df = pd.read_pickle('data/stage1/features_18_extracted.pkl'); df['text'] = df['text'].fillna('')
GCOL = 'gender_label' if 'gender_label' in df.columns else 'gender'
if 'binary_label' not in df.columns:
    df['binary_label'] = (~df['TID'].str.contains('control')).astype(int)
y = df['binary_label'].values.astype(int)
tr_idx, va_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)
log(f"train={len(tr_idx)} val={len(va_idx)}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
texts = df['text'].tolist()

class DS(Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return {'text': texts[j], 'label': int(y[j])}

def collate(batch):
    enc = tok([b['text'] for b in batch], truncation=True, max_length=MAX_LEN,
              padding=True, return_tensors='pt')
    enc['labels'] = torch.tensor([b['label'] for b in batch])
    return enc

train_dl = DataLoader(DS(tr_idx), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=4)
val_dl   = DataLoader(DS(va_idx), batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=4)

# ── model: fp32 master weights (avoids the fp16-config NaN), AMP for speed ───
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, dtype=torch.float32, ignore_mismatched_sizes=True).to(DEVICE)
if NGPU > 1: model = nn.DataParallel(model)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.06*total_steps), total_steps)
scaler = torch.cuda.amp.GradScaler()

@torch.no_grad()
def evaluate():
    model.eval(); ps, ys = [], []
    for enc in val_dl:
        labels = enc.pop('labels')
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(**enc).logits.float()
        ps.append(torch.softmax(logits, -1)[:, 1].cpu().numpy()); ys.append(labels.numpy())
    p, t = np.concatenate(ps), np.concatenate(ys)
    return roc_auc_score(t, p), f1_score(t, (p > 0.5).astype(int))

best_auc = -1
for ep in range(EPOCHS):
    model.train()
    for step, enc in enumerate(train_dl):
        labels = enc.pop('labels').to(DEVICE)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        opt.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(**enc, labels=labels)
            loss = out.loss.mean()                       # DataParallel -> vector
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 100 == 0: log(f"  ep{ep} step{step}/{len(train_dl)} loss={loss.item():.4f}")
    auc, f1 = evaluate(); log(f"epoch {ep}: val AUC={auc:.4f} F1={f1:.4f}")
    if auc > best_auc:
        best_auc = auc
        (model.module if NGPU > 1 else model).save_pretrained(FT_DIR); tok.save_pretrained(FT_DIR)
        log(f"  ↳ saved best (AUC={auc:.4f})")

assert best_auc > 0.70, f"SANITY GATE FAILED: best val AUC {best_auc:.3f} <= 0.70"
log(f"FINE-TUNE DONE. best val AUC={best_auc:.4f}")

# ── extract FT [CLS] (encoder body, last_hidden[:,0]) for all posts ──────────
del model; torch.cuda.empty_cache()
enc_model = AutoModel.from_pretrained(FT_DIR, output_hidden_states=True).to(DEVICE).eval()

@torch.no_grad()
def extract(layer=None, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt').to(DEVICE)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            o = enc_model(**e)
        h = o.hidden_states[layer] if layer is not None else o.last_hidden_state
        out.append(h[:, 0, :].float().cpu().numpy())
    return np.vstack(out)

cls = extract()
np.save(f'{OUT}/cls_embeddings.npy', cls); log(f"saved cls_embeddings {cls.shape}")

# ── probing ──────────────────────────────────────────────────────────────────
def targets():
    t = {f: (df[f].fillna(0).values > np.median(df[f].fillna(0).values)).astype(int) for f in SHORTCUTS + CONTROLS}
    t['gender'] = (df[GCOL].values == 'female').astype(int)
    t['binary_label'] = y
    return t
T = targets()

def probe_auc(X, yv, folds=5):
    Xs = StandardScaler().fit_transform(X)
    return cross_val_score(LogisticRegression(max_iter=2000), Xs, yv,
                           cv=StratifiedKFold(folds, shuffle=True, random_state=SEED), scoring='roc_auc').mean()

rows = [{'target': k, 'role': ('shortcut' if k in SHORTCUTS else 'control' if k in CONTROLS else k),
         'auc': probe_auc(cls, v)} for k, v in T.items()]
pd.DataFrame(rows).to_csv(f'{OUT}/probing_results.csv', index=False); log("saved probing_results.csv")
for r in rows: log(f"  probe {r['target']:20s} AUC={r['auc']:.4f}")

# layer-wise (3-fold for speed) over shortcuts + gender + label
nL = enc_model.config.num_hidden_layers + 1
lw = []
for L in range(nL):
    h = extract(layer=L)
    for k in SHORTCUTS + ['gender', 'binary_label']:
        lw.append({'layer': L, 'target': k, 'auc': probe_auc(h, T[k], folds=3)})
    log(f"  layerwise L{L} done")
pd.DataFrame(lw).to_csv(f'{OUT}/layerwise_probing.csv', index=False); log("saved layerwise_probing.csv")
log("ALL DONE.")
