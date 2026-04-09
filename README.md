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

### Stage 4 — Debiasing Interventions (planned)

Counterfactual data augmentation, adversarial debiasing, loss reweighting, Product-of-Experts ensembles.

### Stage 5 — Evaluation (planned)

Cross-dataset generalization, fairness metrics (equalized odds, demographic parity), shortcut suppression metrics.

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