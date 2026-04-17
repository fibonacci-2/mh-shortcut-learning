# Shortcut Learning Audit for Mental Health NLP Models

**Auditing gender-correlated shortcut features in transformer-based mental health classifiers.**

---

## Objective

Test whether transformer models (DistilRoBERTa, MentalBERT) rely on gender-correlated writing-style shortcuts — rather than clinical signal — when classifying mental health risk from social media text. Identify, quantify, and (in later stages) mitigate this reliance.

## Data

| Dataset | Source | Posts | Label scheme |
|---------|--------|------:|-------------|
| CSSRS | Reddit (r/SuicideWatch + controls) | 6,773 | supportive / indicator / ideation / behavior / attempt |
| Mindset | Reddit (mental health subs) | 19,200 | has_condition / no_condition |
| UMD | Reddit (CLPsych shared task) | 21,264 | at_risk / no_risk |
| **Total** | | **47,237** | Binary: positive (80%) / negative (20%) |

Gender labels obtained via 7-tier regex self-disclosure matching → propagated by user ID → 16,306 posts (519 users, 34.5% coverage; M:F ≈ 3:1).

## Pipeline

```
Stage 0  Gender Annotation        →  00_gender_annotation.ipynb, 00b_gender_inference.ipynb
Stage 1  Shortcut Extraction      →  01_shortcut_candidate_extraction.ipynb
Stage 2  Shortcut Auditing        →  02_shortcut_auditing.ipynb
Stage 3  Causal Disentanglement   →  03_causal_disentanglement.ipynb
Stage 4  Debiasing Interventions  →  04_debiasing_interventions.ipynb  (planned)
Stage 5  Evaluation               →  05_evaluation.ipynb               (planned)
```

## Methods & Results

### Stage 0 — Gender Annotation ✅

7-tier regex pattern matching for explicit gender self-disclosure (e.g., *"I'm a 24F"*, *"as a man"*). 887 direct matches propagated to 16,306 posts across 519 users. LIWC-22 applied externally to produce 132-column annotated dataset.

### Stage 1 — Shortcut Candidate Extraction ✅

Engineered 14 gender-associated linguistic features (hedge density, pronoun ratios, LIWC categories, etc.). Standalone AUC all < 0.60; combined = 0.568.

**MI decomposition** flagged 3 features with > 40% mutual-information drop when conditioning on gender:

| Feature | I(f; gender) | MI drop | Association |
|---------|-------------|---------|-------------|
| question_density | 0.0060 | 43% | Feminine-coded |
| negative_emotion | 0.0059 | 47% | Both (depression + feminine) |
| certainty | 0.0044 | 42.5% | Masculine-coded |

### Stage 2 — Shortcut Auditing ✅

**Probing classifiers** on frozen DistilRoBERTa [CLS] embeddings (768-d):

| Probe target | AUC | R² |
|-------------|-----|-----|
| question_density | 0.979 | 0.821 |
| negative_emotion | 0.889 | 0.248 |
| certainty | 0.855 | 0.142 |
| gender (m/f) | 0.766 | — |
| binary label | 0.722 | — |

**SHAP attribution** (3-shortcut logistic model, AUC=0.556): `negative_emotion` dominates (81.6% of total |SHAP|).

**Feature ablation**: Removing all 3 shortcuts from 14-feature model → ΔAUC = +0.008 (1.3%). Gender-stratified: all 3 features affect female predictions more than male.

### Stage 3 — Causal Disentanglement ✅

**Causal mediation** (Baron-Kenny, 1000 bootstrap): Only `negative_emotion` has a significant indirect effect (ACME = +0.006, 95% CI excludes zero), but acts as a **suppressor** (−6.6% proportion mediated — counteracts gender bias rather than amplifying it). `certainty` and `question_density`: n.s.

**INLP concept erasure** (Ravfogel et al., 2020): Removing shortcut directions from [CLS] embeddings:
- Concept AUC drops substantially (−0.05 to −0.08) → erasure works
- Gender probe AUC barely changes (Δ < −0.002) → shortcuts do **not** mediate gender encoding
- Label probe AUC unchanged (|Δ| < 0.002) → model does **not** causally rely on shortcuts for predictions

**Counterfactual flips**: Erasing all 3 shortcuts flips only 2.6% of predictions (3.5% female vs 2.3% male — consistent gender asymmetry).

**Conditional independence tests**: All three LR tests reject Y ⊥ G | S (p ≈ 0). Gender carries information about labels **beyond** what shortcuts explain. Adding gender to 14 features raises AUC by +0.041.

**Key takeaway**: The model *encodes* shortcut features but does *not causally rely* on them. Gender information flows through pathways other than these 3 shortcuts. Debiasing shortcuts alone is insufficient.

### Stage 4 — OOD Evaluation ✅  (`06_ood_evaluation.ipynb`)

Three experiments quantify how gender-mediated shortcuts behave under distribution shift.

#### Exp 1: Cross-Dataset Transfer (Mindset → CSSRS / UMD)

Train Full (18), No-shortcut (12), and Shortcut-only (6) logistic probes on Mindset; test OOD.

| Model | ID AUC | CSSRS OOD | UMD OOD |
|---|---|---|---|
| Full (18) | 0.699 | 0.586 | 0.594 |
| No-shortcut (12) | 0.661 | 0.503 | **0.595** |
| Shortcut-only (6) | 0.678 | **0.615** | 0.574 |

User-level AUC (aggregating per-user): CSSRS shortcut-only = **0.720**; UMD models converge (~0.77–0.80).

**Result:** On UMD, shortcut-only is worst and removing shortcuts loses nothing (0.595 ≈ 0.594) — the predicted confound pattern. On CSSRS, shortcut features happen to align with that dataset's structure.

#### Exp 2: Gender-Balanced Resampling (Mindset-internal)

Downsample so P(label=1|F) = P(label=1|M) = 16.9%, recompute MI.

| Feature group | MI drop (original) | MI drop (balanced) | Δ |
|---|---|---|---|
| 6 shortcuts | 42.5% | −3.6% | **−46.0 pp** |
| 4 controls | 24.9% | −12.3% | −37.2 pp |

**Result:** Shortcut MI drops collapse to near-zero once the gender–label confound is removed.

#### Exp 3: Cross-Gender Transfer (Mindset-internal)

Train on one gender, test on the other. Transfer is symmetric and base-rate dominated. Shortcut model shows the largest cross-gender degradation (M→F: +0.024, F→M: −0.023).

#### Key Conclusions

1. Shortcut MI drops collapse under gender balancing (42.5% → −3.6%).
2. On UMD, shortcuts are pure confounds: removing them loses no OOD signal.
3. Cross-gender transfer is symmetric; shortcuts degrade most.

All results saved under `data/evaluation/` (14 CSV + PNG files).

## Directory Structure

```
data/
├── stage0/          Gender annotation, standardized datasets, LIWC output
│   ├── standardized/          3 raw standardized CSVs
│   ├── gender-annotated/      Gender-labeled PKL + per-dataset CSVs
│   ├── mindset-manual-annot/  Gold-standard manual annotations
│   ├── all-annotated.csv      16,306 × 132 LIWC-annotated posts
│   └── gender_inspection_sample_100.csv
├── stage1/          Shortcut candidate extraction outputs
│   ├── features_14_extracted.pkl
│   ├── shortcut_14_feature_aucs.csv
│   ├── shortcut_14_mi_analysis.csv
│   └── *.png (feature AUC + MI visualizations)
├── stage2/          Shortcut auditing outputs
│   ├── cls_embeddings_16306.npy   Frozen [CLS] embeddings (768-d)
│   ├── stage2_probing_results.csv
│   ├── stage2_ablation_results.csv
│   ├── stage2_shap_values.npy
│   └── *.png (probing, SHAP, ablation visualizations)
└── stage3/          Causal disentanglement outputs
    ├── stage3_mediation_results.csv
    ├── stage3_concept_erasure_results.csv
    ├── stage3_counterfactual_flips.csv
    └── *.png (DAG, mediation, erasure visualizations)
```

## Environment

- Python 3.13.3, PyTorch 2.6.0+cu124, 4× Tesla V100-SXM2-16GB
- transformers 5.5.0, shap 0.48.0, scikit-learn 1.6.1, fairlearn 0.13.0


# Us vs GAudit

## How Each Identifies Shortcuts

| Dimension | G-AUDIT | Your MI Decomposition |
|---|---|---|
| **Core idea** | 2D scatter: *Utility* $MI(A; Y)$ × *Detectability* $MI(A; \hat{A})$ where $\hat{A} = f(X)$ | Compare $I(f; Y)$ vs $I(f; Y \mid G)$; a large **drop** when conditioning on gender flags a shortcut |
| **What counts as a shortcut** | High utility **AND** high detectability (upper-right quadrant) | Feature whose predictive MI substantially decreases once gender is partialled out |
| **Confound variable** | Any metadata attribute $A$ (age, sex, site, device…) | Specifically gender $G$ |
| **Role of a model** | Trains a *surrogate* $f: X \to \hat{A}$ to measure detectability; utility is model-free | Entirely model-free at the identification stage (MI computed from discrete feature × label × gender tables) |

---

## Rigour & Methodological Soundness

### 1. Causal reasoning

**G-AUDIT** explicitly considers the *direction* of the data-generating process ($X \to Y$ causal vs $Y \to X$ anti-causal) and adjusts whether it conditions on $Y$ when estimating detectability, to avoid collider bias. This is a genuine strength — it means detectability estimates are less likely to be inflated by information leaking through the label.

**Your approach** implicitly assumes the anti-causal direction ($Y, G \to \text{features}$) and conditions on $G$ to isolate spurious from genuine signal. You don't need to worry about collider bias because $G$ is a confounder (common cause of feature and label via base-rate disparity), not a collider. The conditioning is therefore *correct* under your DAG, but the DAG itself is assumed rather than tested.

**Verdict:** Both are causally principled. G-AUDIT is more general (handles either direction); your approach is more targeted and arguably more interpretable for the specific gender-confounding scenario.

### 2. What the test actually measures

**G-AUDIT's utility** $MI(A; Y)$ answers: *"Is this attribute statistically associated with the label?"* This is identical to your unconditional $I(f; Y)$ — both are marginal MI. The novelty in G-AUDIT is *detectability*: can the model actually *see* the attribute in the raw input? If not, even high utility isn't risky. This is a practically important insight — a confound that's invisible to the model is harmless.

**Your MI drop** $I(f; Y) - I(f; Y \mid G)$ answers a *different* question: *"How much of this feature's predictive power is mediated by gender?"* This directly targets the **confounding mechanism** rather than just flagging co-occurrence. A feature can have high utility and high detectability in G-AUDIT but still be a *legitimate* predictor if it's causally related to the label independently of the protected attribute — G-AUDIT can't distinguish this without domain expertise, whereas your conditional MI explicitly tests for it.

**Verdict:** Your approach is **more specific** to the confounding question. G-AUDIT is **more general** but requires post-hoc domain expertise to separate genuine predictors from shortcuts.

### 3. Detectability gap

G-AUDIT adds a dimension your pipeline lacks at the identification stage: whether the model can actually *recover* the attribute from raw input $X$. You address this *later* (Stage 2 probing classifiers), but G-AUDIT bakes it into the initial ranking. This is methodologically clean — it prevents wasting effort auditing features the model can't exploit.

However, your probing analysis (Stage 2) is arguably *richer*: you probe at every layer, measure R², AUC, and track depth trajectories. G-AUDIT uses a single surrogate model and reports a scalar detectability. Your layer-wise analysis reveals *how* the model encodes shortcuts, not just *whether* it can.

### 4. Statistical estimation

G-AUDIT uses chance-adjusted MI (Vinh et al., 2009/2010) to handle categorical variables with different cardinalities. Your approach uses standard binned MI estimates. For your setting (binary label, binary gender, continuous features discretised at median), chance adjustment matters less because all variables are low-cardinality. But for G-AUDIT's more heterogeneous attribute types (age bins with 18 classes, etc.), the correction is important.

### 5. Worst-case performance bounding

G-AUDIT includes a **synthetic calibration** step: inject a fully-detectable synthetic shortcut at varying utility levels, measure the resulting AUC drop on a counterfactual test set. This translates MI values into familiar performance metrics. Your pipeline doesn't do this — you instead measure *actual* AUC degradation via ablation (Stage 2C) and OOD transfer (Stage 4), which is more direct but requires more computation.

---

## Summary

| Criterion | G-AUDIT | Your Approach |
|---|---|---|
| **Generality** | ✅ Modality-agnostic, any attribute | Targeted to gender × MH |
| **Causal direction handling** | ✅ Explicit causal/anti-causal | Assumes one DAG |
| **Confound-specificity** | ❌ Can't separate genuine from spurious without domain knowledge | ✅ Conditional MI directly tests confounding |
| **Detectability at identification** | ✅ Built-in | Deferred to probing (Stage 2) |
| **Mechanistic depth** | Scalar per attribute | ✅ Layer-wise, SHAP, ablation |
| **Performance impact** | Synthetic worst-case bound | ✅ Actual ablation + OOD transfer |
| **Statistical assumptions** | Chance-adjusted MI | Standard binned MI (sufficient for low-cardinality) |

**Bottom line:** G-AUDIT is a broader, more portable *screening tool* — it efficiently ranks many attributes across modalities but stops at "this attribute is risky." Your MI decomposition is a *deeper diagnostic* — it directly isolates the confounding mechanism ($I(f;Y) - I(f;Y|G)$) and then follows through with encoding analysis, attribution, ablation, and OOD validation. For a focused study on gender shortcuts in mental health NLP, your approach is **more methodologically rigorous** because it answers the causal question directly rather than flagging correlations that require human interpretation. G-AUDIT would be the better choice for a quick first-pass audit across dozens of attributes before you know which ones matter.