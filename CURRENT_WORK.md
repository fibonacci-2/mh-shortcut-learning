# Current Work — Stage 2 Multi-Model Robustness Sweep

> Living status doc for the in-flight work. The methodology, datasets, and the
> completed Stage 0–4 findings live in [`README.md`](README.md); this file tracks
> **what is running, done, and blocked right now**. Last updated: 2026-05-24.

## Goal

Stage 2 (shortcut auditing) was originally run on a single backbone (frozen
`distilroberta-base`). The current work **replicates the full probing → SHAP →
ablation pipeline across four backbones**, each in a *frozen* and (where
applicable) *fine-tuned* variant, to test whether the headline Stage 3 finding —
*the model **encodes** gender shortcuts but does **not causally rely** on them* —
is an artifact of one architecture/training regime or holds generally.

Each variant lives in its own notebook and writes to its own `data/stage2_*`
directory so results can be compared side by side. These per-model results feed
the robustness section of `manuscript.tex` (which already references MentalRoBERTa
and DeBERTa).

## Status matrix

| # | Notebook | Backbone | Frozen | Fine-tuned | Output dir |
|---|----------|----------|:------:|:----------:|------------|
| 02  | `02_shortcut_auditing.ipynb`               | `distilroberta-base`        | ✅ done | — (n/a) | `data/stage2/` |
| 02b | `02b_shortcut_auditing_finetuned.ipynb`    | `distilroberta-base` (FT)   | — | ✅ outputs present¹ | `data/stage2_finetuned/` |
| 02c | `02c_shortcut_auditing_mentalroberta.ipynb`| `mental/mental-roberta-base`| ✅ done | ✅ done | `data/stage2_mentalroberta/{frozen,finetuned}/` |
| 02d | `02d_shortcut_auditing_deberta.ipynb`       | `microsoft/deberta-v3-base` | ✅ done | 🔴 **in progress / blocked** | `data/stage2_deberta/{frozen,finetuned}/` |

¹ `02b` outputs (embeddings, probing, SHAP, ablation) are present and recent
(2026-05-23), but its `trainer.train()` cell currently shows the v5 `on_train_begin`
error (see Known issues). The existing results were produced from a fine-tuned
model saved on an earlier run (`data/stage2_finetuned/ft_model/`); re-running the
fine-tune from scratch will hit that bug until it is fixed.

## Active task — finish `02d` (DeBERTa-v3 fine-tuned)

This is the only incomplete cell of the sweep.

- `data/stage2_deberta/frozen/` is fully populated; `data/stage2_deberta/finetuned/`
  is **empty** — the fine-tuned half has not produced outputs yet.
- The notebook is partially executed (~12/25 code cells); the blocker is the
  fine-tuning cell.
- **Fixed (2026-05-24):** that cell raised
  `TypeError: ... unexpected keyword argument 'group_by_length'`. transformers v5
  renamed the flag — changed `group_by_length=True` →
  `train_sampling_strategy='group_by_length'`. The cell now instantiates
  `TrainingArguments` cleanly.

Remaining steps to close out `02d`:

1. Run the (now-fixed) fine-tuning cell. ⚠️ Watch for the `on_train_begin` error
   below — it will likely fire next, since the same `eval_strategy='epoch'`
   pattern is what trips it in `02b`/`02c`.
2. After training: extract fine-tuned `[CLS]` embeddings → probing → SHAP →
   ablation, populating `data/stage2_deberta/finetuned/`.
3. Confirm fine-tune sanity gate passes (`best eval_AUC > 0.70`) before trusting
   downstream embeddings.

## Known issues (transformers v5.5.0 migration)

The `.venv` kernel was upgraded to **transformers 5.5.0**, which removed/renamed
several `TrainingArguments` options. Fallout, in order of severity:

- 🔴 **`on_train_begin must be called before on_evaluate`** — raised by
  `trainer.train()` in `02b` (cell idx 3) and `02c` (cell idx 15); not yet hit in
  `02d` only because training hadn't reached that point. A v5 callback-ordering
  regression triggered by per-epoch evaluation. **Not yet fixed** — this is the
  next thing to resolve to make the fine-tunes reproducible from scratch.
- ✅ **`group_by_length`** → `train_sampling_strategy='group_by_length'`. Fixed in
  `02d`; apply the same rename to `02b`/`02c` if re-running them.
- ⚠️ **`warmup_ratio` deprecation** — still works in 5.5.0, slated for removal in
  v5.2; switch to `warmup_steps` eventually. Non-blocking.
- ℹ️ **`02c`** also shows `name 'probe_df_frozen' is not defined` (cells 13/26) —
  out-of-order execution; re-run the frozen-probe cell that defines it before the
  cells that consume it.

> Note: the system `python3` on PATH (`/c1/apps/python3/3.13.3`) has transformers
> **4.57.3**, where `group_by_length` is still valid — so config errors may not
> reproduce outside the kernel. Always validate against the notebook kernel.

## Environment / how to run

- **Kernel:** `.venv` → `./.venv/bin/python` (the kernelspec named `.venv`).
  *Not* the system `python3`.
- Python 3.13.3, PyTorch 2.6.0+cu124, 4× Tesla V100-SXM2-16GB.
- transformers **5.5.0**, shap 0.48.0, scikit-learn 1.6.1, fairlearn 0.13.0.
- **DeBERTa-v3 must load as fp32** (`dtype=torch.float32`); its config defaults to
  fp16, which NaNs under AdamW with no GradScaler. The `.failed_run_*` dirs under
  `data/stage2_deberta/` are artifacts of that earlier failure. `fp16=True` (AMP)
  is safe only because the master weights are fp32.

## Output layout (per variant)

Each model variant produces the same artifact set (names vary slightly by notebook):

```
<variant_dir>/
├── cls_embeddings*.npy              frozen / fine-tuned [CLS] embeddings
├── probing_results.csv  + .png      per-shortcut probe AUC / R²
├── layerwise_probing.csv + heatmap  depth-resolved probing
├── shap_values.npy + shap_*.png     SHAP attribution
├── ablation_results.csv + .png      shortcut-removal ΔAUC
├── gender_stratified_ablation.csv   ablation split by gender
└── ft_model/                        saved fine-tuned checkpoint (FT variants)
```
