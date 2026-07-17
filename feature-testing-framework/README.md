# mh-shortcut-mi

Minimal, dataset-agnostic pipeline for the two core pieces of the shortcut
audit: gender/clinical linguistic feature extraction and the conditional
mutual-information test (`I(f;Y)` vs `I(f;Y|G)`) that flags gender-mediated
shortcuts.

## Layout

- `extract_gender.py` — **stage 0**, runs before everything else. Regex-based
  self-disclosure gender extraction (e.g. `"(24F)"`, `"I'm 32m"`, `"28m here"`)
  from raw text, at user level. Produces the `gender` column every dataset
  loader below expects.
- `features.py`  — the unified 18-feature set: Hyland hedge lexicon + custom
  lexicons + real LIWC-22 output (see `merge_liwc.py`) + one Empath category
  (`pain`, the one construct LIWC's default dictionary doesn't cover). Three
  tiers, see `FEATURE_INFO`: `shortcut_candidate` (12, hypothesized
  gender-coded style), `clinical_reference` (2, content that may legitimately
  vary by condition), `gendered_reference` (4, validated real LIWC gender
  differences — sanity-checks the test still finds genuine signal). Only
  `synthetic` lacks LIWC/Empath coverage (smoke-test fixture, expected).
- `empath_features.py` — full 194-category Empath sweep, for exploratory runs
  (`--feature-set empath`/`both`); superseded by `features.py` as the default
  battery now that every real dataset has genuine LIWC-22 coverage.
- `prepare_liwc_batch.py` / `merge_liwc.py` — one-off pair: batch every
  dataset's raw text into one CSV for an external LIWC-22 CLI run, then split
  the annotated output back onto each dataset's raw CSV by row position.
- `mi_test.py`   — MI decomposition + gender-permutation significance test
  (Benjamini-Hochberg FDR across the battery), **plus an effect-size floor**:
  `shortcut = (perm_q < fdr) & (MI_drop_pct >= MIN_DROP_PCT)` (default 50%).
  perm_q alone conflates statistical detectability with practical effect size
  — at large n even a genuine, gender-independent feature's small MI drop can
  read as "significant" (confirmed via the `synthetic` ground-truth check).
- `datasets/`    — one loader module per corpus; each exposes `load(path)` and
  `conditions(df)` and returns a DataFrame with columns `text`, `label`
  (condition name, `'control'` for the control group), `gender`
  (`'male'`/`'female'`), plus any LIWC columns already annotated (`Affect`,
  `Social`, `emo_neg`, ... — missing ones default to 0).
- `run.py`       — CLI: load a dataset, compute features (`--feature-set
  core|empath|both`), run the MI test per condition, print + save results.

## Usage

```
pip install -r requirements.txt

# stage 0: annotate raw text with self-disclosed gender (skip if your dataset
# already has a gender field, e.g. MIMIC's patients.gender)
python extract_gender.py --input data/raw/depression.csv --out-dir data/gender

# stage 1: feature extraction + MI test
python run.py --dataset synthetic                                          # smoke test, no data needed
python run.py --dataset mindset --data data/strong-ann.csv
python run.py --dataset mindset --data data/strong-ann.csv --condition depression --out results.csv
python run.py --dataset umd --data data/gender/umd-raw.csv --out results/umd.csv
python run.py --dataset umd_crowd --data data/gender/umd-crowd-raw.csv --out results/umd_crowd.csv

# broader feature sweep (no LIWC needed) + stricter significance for a bigger battery
python run.py --dataset umd_demographics --feature-set empath --fdr 0.05
python run.py --dataset umd_demographics --feature-set both --out results.csv
```

`extract_gender.py` only matches **explicit self-disclosures** (age token
co-occurring with a gender token, e.g. `"(24F)"`, `"I'm 32m"`, `"28m here"`) —
the high-precision pattern used for the Mindset corpus. It does not fall back
to weaker contextual gender mentions, so every non-null `gender` in its output
is a high-confidence disclosure. Quoted/reported speech is stripped first so
disclosures inside quotes aren't attributed to the wrong person. Gender is
resolved per `user_id` (first disclosure found across any of their posts) and
applied to all of that user's posts. Output: one annotated CSV per input file
plus a `summary.json` (counts, gender distribution, output paths) in
`--out-dir`, and the same summary printed to console.

## Findings so far

Ran the unified feature set + effect-size-floored MI test across all 12
registered datasets (22 conditions).

| dataset | conditions flagged | notes |
|---|---|---|
| `synthetic` | 1/1 (ground truth) | engineered shortcut (`fp_singular`, 86% drop) correctly flagged; engineered genuine signal (`negative_emotion`, 7.5% drop) correctly *not* flagged |
| `mimic_prostate_cancer` | 1/1 | positive control (anatomically male) — expected |
| `mimic_ovarian_cyst` | 1/1 | positive control (anatomically female) — expected |
| `mimic_cad`, `mimic_osteoporosis`, `mimic_autism` | 0/3 | real audit targets — clean |
| `umd`, `umd_crowd`, `umd_demographics` | 0/9 | clean |
| `irf_belong`, `irf_burden` | skipped | only ~1% of rows have a gender label at all (n too small) |
| `mindset` | 6/7 | **the one real finding** — see below |

**The pipeline validates correctly**: both positive controls (anatomically
sex-determined conditions) and the synthetic ground truth behave exactly as
designed, confirming the MI test + effect-size floor distinguish genuine
shortcuts from incidental correlation rather than flagging everything gendered.

**`mindset` is the one dataset with a real, credible shortcut**: 6 of 7
conditions (all but depression) flag at least one feature, `fp_singular`
(first-person-singular rate) in 5 of them. Root cause, confirmed from the
data itself, not assumed:
1. Every condition skews clearly female (M:F as low as 0.27:1) while control
   skews male (1.45:1) — gender and label are confounded by real Reddit
   self-selection into these subreddits, in *consistent, opposite* directions
   per condition. `umd`/`umd_crowd` have gender skew too, but same-direction
   or inconsistent across risk tiers, so it doesn't add up to the same effect.
2. mindset's authors *are* the gendered subjects (Reddit self-posts), so their
   own writing style directly reflects their own gender — unlike MIMIC, where
   a clinician (not the patient) writes the note.
3. `pain` in mindset's ocd/ptsd (92%+ drop) is a second, smaller signal worth
   a qualitative look — surprising since `pain` is a clinical-reference
   feature, not a style feature.

**Caveats to keep in mind:**
- `mimic_autism` (n=252) and `irf_*` (skipped) are underpowered — a clean
  result there means "undetectable," not "absent."
- Several "control" groups aren't neutral populations (`umd_demographics`'
  `no_risk` is still SuicideWatch-adjacent; `irf`'s control is
  same-population) — read clean results with that in mind.
- `MIN_DROP_PCT=50` and `fdr=0.05` are reasoned defaults, not the only
  defensible choice — `MI_drop_pct` itself (continuous) is more informative
  than the binary flag for borderline cases.

## Adding a dataset

Drop a new file in `datasets/`, e.g. `datasets/mimic.py`, implementing:

```python
def load(path) -> pd.DataFrame:     # columns: text, label, gender [, LIWC cols]
def conditions(df) -> list[str]:    # condition names to test (excludes 'control')
```

then register it in `datasets/__init__.py`. Nothing else changes.
