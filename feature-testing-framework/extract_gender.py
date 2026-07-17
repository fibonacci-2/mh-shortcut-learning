#!/usr/bin/env python3
"""Self-disclosure gender extraction — runs early in the pipeline, before
feature extraction, to produce the gender label every downstream dataset
loader consumes.

Every match here is still a direct, self-referential textual claim (never
inferred from writing style), so it stays valid as an independent "G" for the
gender-shortcut MI test -- but the *convention* people use to state it differs
by platform, so patterns are grouped into confidence tiers rather than one
flat list:

  high   — age_gender:   "(24F)", "I'm 32m", "28m here"     (Reddit convention)
           pronoun_tag:   "she/her", "he/him", "they/them"   (Twitter/bio convention)
  medium — self_id:       "I'm a woman", "as a man", "biologically female"
           parental:      "mom of 3", "proud father of two"

A user's gender is the highest-confidence match found across any of their
posts (ties broken by whichever is found first), applied to all of their
posts. Downstream loaders decide their own confidence cutoff (see
datasets/umd_demographics.py, which keeps only 'high').

Usage:
    python extract_gender.py --input data/raw/depression.csv --out-dir data/gender
    python extract_gender.py --input data/raw/*.csv --out-dir data/gender
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── tier 1 (high): age token co-occurring with a gender token, Reddit-style ──
_SELF_REF = (
    r"(?:i\s*['‘’ʼ´`]?\s*m"
    r"|i\s+am"
    r"|im(?=\s+\d)"
    r"|me(?=\s+\d)"
    r"|hi[,!]?"
    r"|hello[,!]?"
    r"|hey[,!]?)"
)
_SELF_DISCLOSE = (
    r"(?:here\b"
    r"|and\s+i(?:'?(?:m|ve|am)|\s+am)\b"
    r"|looking\s+for\b"
    r"|seeking\b"
    r"|single\b"
    r"|married\b"
    r"|asking\s+for\b"
    r"|need(?:ing)?\s+(?:some\s+)?(?:advice|help)\b"
    r"|new\s+to\b)"
)
_GENDER_TOKEN = r"(m|f|nb|male|female|nonbinary|enby)"

AGE_GENDER_PATTERNS = [
    # (34F), [28m], (23/f), (23-F)
    re.compile(rf"[\(\[]\s*(\d{{1,3}})\s*[/\-,]?\s*{_GENDER_TOKEN}\s*[\)\]]", re.I),
    # "i'm 24m", "i am 32f", "im 28m", "me 24m", "hi, 32f"
    re.compile(rf"\b{_SELF_REF}\s+(?:a\s+|an\s+)?(\d{{1,3}})\s*[/\-,]?\s*{_GENDER_TOKEN}\b", re.I),
    # "32F here", "28m, looking for", "24m single"
    re.compile(rf"(?<![.\d$£€¥])\b(\d{{1,3}})\s*[/\-,]?\s*{_GENDER_TOKEN}\b"
               rf"\s*[,!.]?\s+{_SELF_DISCLOSE}", re.I),
]

# ── tier 2 (high): pronoun self-tag, Twitter/bio convention ─────────────────
PRONOUN_TAG_PATTERNS = [
    re.compile(r'\b(she|he|they)\s*/\s*(?:her|him|them)(?:\s*/\s*(?:hers|his|theirs))?\b', re.I),
]

# ── tier 3 (medium): direct self-identification phrase ──────────────────────
SELF_ID_PATTERNS = [
    re.compile(r"\b(?:i'?m|i\s+am)\s+(?:a|an)\s+(woman|man|girl|boy|female|male|guy|gal|dude|lady)\b", re.I),
    re.compile(r"\bas\s+an?\s+(woman|man|girl|boy|female|male)\b", re.I),
    re.compile(r"\bbiologically\s+(male|female)\b", re.I),
]

# ── tier 4 (medium): parental self-reference ─────────────────────────────────
PARENTAL_PATTERNS = [
    re.compile(r"\b(?:single|proud|first[- ]time|new)?\s*(mom|mommy|mother|dad|daddy|father)\s+(?:of|to)\b", re.I),
]

GENDER_MAP = {
    'm': 'male', 'male': 'male', 'man': 'male', 'boy': 'male', 'guy': 'male', 'dude': 'male',
    'he': 'male', 'dad': 'male', 'daddy': 'male', 'father': 'male',
    'f': 'female', 'female': 'female', 'woman': 'female', 'girl': 'female', 'gal': 'female', 'lady': 'female',
    'she': 'female', 'mom': 'female', 'mommy': 'female', 'mother': 'female',
    'nb': 'nonbinary', 'nonbinary': 'nonbinary', 'enby': 'nonbinary', 'they': 'nonbinary',
}

# (patterns, confidence, pattern_type, has_age) in priority order
_TIERS = [
    (AGE_GENDER_PATTERNS, 'high', 'age_gender', True),
    (PRONOUN_TAG_PATTERNS, 'high', 'pronoun_tag', False),
    (SELF_ID_PATTERNS, 'medium', 'self_id', False),
    (PARENTAL_PATTERNS, 'medium', 'parental', False),
]
_CONF_RANK = {'high': 2, 'medium': 1}


def _valid_age(token):
    return 10 <= int(token) <= 100


def extract_from_text(text):
    """Return (age, gender, matched_excerpt, confidence, pattern_type) for the
    first disclosure found in `text` (tiers checked in priority order), or
    all-None if there isn't one."""
    text = '' if pd.isna(text) else str(text)
    text = re.sub(r'"[^"]*"', ' ', text)          # strip quoted speech
    text = re.sub(r'(?m)^[ \t]*>.*$', ' ', text)   # strip quote-reply lines

    for patterns, confidence, pattern_type, has_age in _TIERS:
        for pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            gender_token = m.group(2 if has_age else 1).lower()
            gender = GENDER_MAP.get(gender_token)
            if has_age and not _valid_age(m.group(1)):
                continue
            age = int(m.group(1)) if has_age else None
            return age, gender, m.group(0), confidence, pattern_type
    return None, None, None, None, None


def extract_user_gender(texts):
    """Highest-confidence disclosure found across any of a user's posts wins.
    Returns the specific post text that produced it too (`source_text`) --
    a user can have many posts, and only one of them contains the match."""
    best = (None, None, None, None, None)
    best_rank = -1
    source_text = None
    for text in texts:
        result = extract_from_text(text)
        rank = _CONF_RANK.get(result[3], -1)
        if rank > best_rank:
            best, best_rank, source_text = result, rank, text
            if rank == max(_CONF_RANK.values()):
                break
    return (*best, source_text)


def process_file(input_csv, out_dir, user_col='user_id', text_col='text'):
    df = pd.read_csv(input_csv)
    df[text_col] = df[text_col].fillna('')

    rows = []
    groups = df.groupby(user_col)[text_col].apply(list)
    for uid, texts in tqdm(groups.items(), total=len(groups), desc=Path(input_csv).stem):
        age, gender, excerpt, confidence, pattern_type, source_text = extract_user_gender(texts)
        rows.append({user_col: uid, 'age': age, 'gender': gender, 'matched_excerpt': excerpt,
                     'confidence': confidence, 'pattern_type': pattern_type,
                     'gender_source_text': source_text if excerpt is not None else None})
    user_demo = pd.DataFrame(rows)

    out = df.merge(user_demo, on=user_col, how='left')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(input_csv).name
    out.to_csv(out_path, index=False)

    n_users = df[user_col].nunique()
    n_gendered = int(user_demo['gender'].notna().sum())
    return {
        'file': Path(input_csv).name,
        'n_posts': len(df),
        'n_users': n_users,
        'n_users_gendered': n_gendered,
        'pct_users_gendered': round(100 * n_gendered / n_users, 1) if n_users else 0.0,
        'gender_counts': user_demo['gender'].value_counts().to_dict(),
        'confidence_counts': user_demo['confidence'].value_counts(dropna=False).to_dict(),
        'pattern_type_counts': user_demo['pattern_type'].value_counts(dropna=True).to_dict(),
        'output_csv': str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', nargs='+', required=True, help='raw CSV(s) with user_id, text columns')
    ap.add_argument('--out-dir', default='data/gender', help='directory for annotated CSVs + summary.json')
    ap.add_argument('--user-col', default='user_id')
    ap.add_argument('--text-col', default='text')
    args = ap.parse_args()

    stats = [process_file(f, args.out_dir, args.user_col, args.text_col) for f in args.input]

    summary_path = Path(args.out_dir) / 'summary.json'
    summary_path.write_text(json.dumps(stats, indent=2))

    print('\n' + '=' * 70)
    for s in stats:
        print(f"{s['file']:30s} users={s['n_users']:6,d}  gendered={s['n_users_gendered']:6,d} "
              f"({s['pct_users_gendered']}%)  {s['gender_counts']}  by_pattern={s['pattern_type_counts']}")
    print(f"\nAnnotated CSVs + summary.json saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
