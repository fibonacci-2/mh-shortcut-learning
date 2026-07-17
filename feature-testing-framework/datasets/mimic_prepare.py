#!/usr/bin/env python3
"""One-off: build data/raw/mimic/mimic-raw.csv from MIMIC-IV's hosp module
(diagnoses_icd, patients, discharge -- gzipped CSVs under data/raw/mimic/).

Flags each admission with a discharge note against 5 ICD-coded conditions
chosen for having a real gender skew, restricted to admissions where the
diagnosis is prominent (seq_num <= MAX_SEQ_NUM) so the note actually
discusses it, not an incidental old code:

  osteoporosis, autism, cad  -- real audit targets: both genders present,
    skewed but not deterministic (matches the literature on diagnostic bias,
    e.g. autism's historical underdiagnosis in women)
  prostate_cancer, ovarian_cyst -- anatomically sex-determined (0 and 3
    counterexamples respectively in this data) -- kept as a POSITIVE CONTROL
    that the pipeline correctly detects genuine biological determinism, not
    a bias finding to be flagged the same way as the other three

Each admission gets a binary flag per condition (not one exclusive label --
an admission can be positive for more than one), so this script keeps the
union of all 5 conditions' admissions plus a bounded random sample of
additional gendered admissions as a shared control pool (processing the
full ~330k-note corpus just for section extraction would be wasteful; each
condition never needs more control rows than its own positive count).

Text is restricted to History of Present Illness + Social History (the
narrative sections reflecting clinician documentation style, not the
templated med/lab/instruction boilerplate filling the rest of a discharge
summary -- MIMIC-IV-Note formats every section as a "Title Case Header:"
line, which is what the section splitter looks for). Explicit gender
words/pronouns are then neutralized -- discharge summaries state sex
directly ("Mrs. ___ is a ___ female...") and use gendered pronouns
pervasively throughout (she/her, not just in an opening sentence), so
without this step the MI test would trivially detect direct lexical
leakage rather than a genuine style-based shortcut. This mirrors standard
counterfactual gender-swapping used in NLP fairness work (e.g. Zhao et al.
2018), just for feature-extraction purposes rather than producing fluent text.

    python datasets/mimic_prepare.py --data-dir data/raw/mimic
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CONDITIONS = {
    'osteoporosis':    {9: ('7330', '7331'),          10: ('M80', 'M81')},
    'prostate_cancer': {9: ('185',),                   10: ('C61',)},
    'ovarian_cyst':    {9: ('6200', '6201', '6202'),   10: ('N830', 'N831', 'N832')},
    'autism':          {9: ('2990',),                  10: ('F840',)},
    'cad':             {9: ('4140',),                  10: ('I25',)},
}
MAX_SEQ_NUM = 3
N_CONTROL_POOL = 30_000  # random non-condition admissions kept as shared control

WANTED_SECTIONS = {'history of present illness', 'social history'}
_SECTION_HEADER = re.compile(r'(?m)^\s*([A-Z][A-Za-z][A-Za-z /\-]{2,60}):\s*$')

# gendered term -> neutral replacement, applied before feature extraction
_GENDER_SWAP = {
    r'\bshe\b': 'they', r'\bhe\b': 'they',
    r'\bher\b': 'their', r'\bhim\b': 'them', r'\bhis\b': 'their', r'\bhers\b': 'theirs',
    r'\bherself\b': 'themself', r'\bhimself\b': 'themself',
    r'\bwoman\b': 'person', r'\bman\b': 'person', r'\bfemale\b': 'person', r'\bmale\b': 'person',
    r'\bmrs\.?\b': '', r'\bmr\.?\b': '', r'\bms\.?\b': '',
    r'\bmother\b': 'parent', r'\bfather\b': 'parent',
    r'\bwife\b': 'spouse', r'\bhusband\b': 'spouse',
    r'\bdaughter\b': 'child', r'\bson\b': 'child',
    r'\bgirl\b': 'person', r'\bboy\b': 'person',
}
_GENDER_SWAP_RE = [(re.compile(pat, re.I), repl) for pat, repl in _GENDER_SWAP.items()]


def de_gender(text):
    for pat, repl in _GENDER_SWAP_RE:
        text = pat.sub(repl, text)
    return text


def extract_sections(text, wanted):
    """Split on MIMIC's 'Title Case Header:' line convention and concatenate
    the bodies of any header in `wanted` (case-insensitive)."""
    text = str(text)
    matches = list(_SECTION_HEADER.finditer(text))
    parts = []
    for i, m in enumerate(matches):
        if m.group(1).strip().lower() not in wanted:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append(text[start:end].strip())
    return '\n'.join(parts)


def code_matches(code, prefixes):
    return str(code).replace('.', '').startswith(prefixes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/raw/mimic')
    ap.add_argument('--out', default='data/raw/mimic/mimic-raw.csv')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    d = Path(args.data_dir)

    print('Loading diagnoses + patients...')
    dx = pd.read_csv(d / 'diagnoses_icd.csv.gz')
    pts = pd.read_csv(d / 'patients.csv.gz', usecols=['subject_id', 'gender'])
    dx = dx.merge(pts, on='subject_id', how='left')
    dx = dx[dx['gender'].isin(['M', 'F'])]
    disch_hadm = set(pd.read_csv(d / 'discharge.csv.gz', usecols=['hadm_id'])['hadm_id'])
    dx = dx[dx['hadm_id'].isin(disch_hadm)]

    print('Flagging admissions per condition...')
    hadm_flags = {}
    for cond, versions in CONDITIONS.items():
        hadms = set()
        for v, prefixes in versions.items():
            sub = dx[(dx['icd_version'] == v) & (dx['seq_num'] <= MAX_SEQ_NUM)]
            mask = sub['icd_code'].apply(lambda c: code_matches(c, prefixes))
            hadms |= set(sub.loc[mask, 'hadm_id'])
        hadm_flags[cond] = hadms
        print(f'  {cond}: {len(hadms):,} admissions')

    positive_any = set().union(*hadm_flags.values())
    hadm_gender = dx.drop_duplicates('hadm_id').set_index('hadm_id')['gender']
    control_candidates = [h for h in hadm_gender.index if h not in positive_any]
    control_sample = pd.Series(control_candidates).sample(
        min(N_CONTROL_POOL, len(control_candidates)), random_state=args.seed)
    target_hadm = positive_any | set(control_sample)
    print(f'Total admissions to extract text for: {len(target_hadm):,} '
          f'({len(positive_any):,} condition-positive + {len(control_sample):,} control)')

    print('Extracting HPI + Social History from discharge notes (chunked)...')
    rows = []
    for chunk in tqdm(pd.read_csv(d / 'discharge.csv.gz', chunksize=20_000,
                                   usecols=['hadm_id', 'text'])):
        chunk = chunk[chunk['hadm_id'].isin(target_hadm)]
        for _, r in chunk.iterrows():
            section_text = extract_sections(r['text'], WANTED_SECTIONS)
            if not section_text:
                continue
            rows.append({
                'hadm_id': r['hadm_id'],
                'gender': {'M': 'male', 'F': 'female'}[hadm_gender[r['hadm_id']]],
                'text': de_gender(section_text),
                **{cond: int(r['hadm_id'] in hadms) for cond, hadms in hadm_flags.items()},
            })

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n{len(out):,} admissions -> {args.out}")
    for cond in CONDITIONS:
        print(f"  {cond}: {out[cond].sum():,} positive")


if __name__ == '__main__':
    main()
