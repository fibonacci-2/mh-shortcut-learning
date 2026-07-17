"""Unified feature set: Hyland hedge lexicon + custom lexicons + real LIWC-22
dictionary output (see merge_liwc.py) + one Empath category LIWC's default
dictionary doesn't cover on its own (pain).

18 features in three tiers (see FEATURE_INFO):
  - shortcut_candidate (12): hypothesized gender-coded STYLE markers -- the
    actual targets of the MI shortcut test (Lakoff 1975; Newman et al. 2008;
    Hyland 1998; Rude, Gortner & Pennebaker 2004).
  - clinical_reference (2): content signal expected to legitimately track a
    condition/its anatomy, not gender style -- sanity-checks the test isn't
    just flagging all clinical content as a "shortcut".
  - gendered_reference (4): LIWC dimensions with real, psychometrically
    validated gender differences (Newman et al. 2008; Kacewicz et al. 2014)
    -- sanity-checks the test still finds genuine gendered signal when it's
    actually there, rather than never triggering at all.

Datasets without real LIWC/Empath coverage (currently just `synthetic`, a
smoke-test fixture) get 0 for every LIWC/Empath-sourced feature here --
expected, not a bug: synthetic only exercises the token-counted features
(fp_singular, fp_plural, hedge_density, excl_intensifiers, question_density,
apology_selfblame) plus its own injected `emo_neg` column.
"""
import re
import numpy as np
import pandas as pd
from empath import Empath

_EMPATH = Empath()

HYLAND_HEDGES_SINGLE = {
    'about', 'almost', 'apparently', 'approximately', 'argue', 'argued', 'argues',
    'around', 'assume', 'assumed', 'assumes', 'broadly', 'certain', 'claim',
    'claimed', 'claims', 'conceivable', 'conceivably', 'could', 'doubt',
    'doubtful', 'essentially', 'estimate', 'estimated', 'fairly', 'feel',
    'feels', 'felt', 'frequently', 'generally', 'guess', 'hypothesize',
    'hypothesized', 'hypothesizes', 'implication', 'imply', 'indicate',
    'indicated', 'indicates', 'largely', 'likely', 'mainly', 'may', 'maybe',
    'might', 'mostly', 'normally', 'occasionally', 'often', 'partly',
    'perhaps', 'plausible', 'plausibly', 'possible', 'possibly', 'postulate',
    'postulated', 'postulates', 'presumably', 'probable', 'probably', 'quite',
    'rather', 'relatively', 'roughly', 'seem', 'seemed', 'seemingly', 'seems',
    'should', 'sometimes', 'somewhat', 'suggest', 'suggested', 'suggests',
    'suppose', 'supposed', 'supposes', 'suspect', 'suspected', 'suspects',
    'tend', 'tended', 'tends', 'think', 'thinks', 'thought', 'typical',
    'typically', 'uncertain', 'uncertainly', 'unclear', 'unlikely', 'usually',
    'wonder', 'wondered', 'wonders', 'would',
}
HYLAND_HEDGES_MULTI = [
    'kind of', 'sort of', 'more or less', 'to some extent',
    'in general', 'in most cases', 'on the whole', 'for the most part',
    'i think', 'i believe', 'i feel', 'i guess', 'i suppose',
    'it seems', 'it appears', 'in my opinion', 'to my knowledge',
]
FP_SINGULAR = {'i', 'me', 'my', 'myself', 'mine'}
FP_PLURAL = {'we', 'us', 'our', 'ours', 'ourselves'}
INTENSIFIERS = {'so', 'really', 'very'}
APOLOGY_PHRASES = [
    'sorry', "i'm sorry", 'im sorry', 'my fault', 'my bad',
    'i should have', "i shouldn't have", 'i shouldnt have',
    'i apologize', 'i apologise', 'forgive me', 'pardon me',
    "it's my fault", 'its my fault', 'i was wrong',
]

# feature name -> (tier, gender/clinical association) -- tier is one of
# shortcut_candidate / clinical_reference / gendered_reference, see module
# docstring. REFERENCE_FEATURES (both reference tiers) is what run.py flags
# as `is_reference_feature` in results.
FEATURE_INFO = {
    'hedge_density':       ('shortcut_candidate', 'feminine-coded'),
    'fp_singular':         ('shortcut_candidate', 'depression + feminine'),
    'fp_plural':           ('shortcut_candidate', 'masculine-coded'),
    'social_relational':   ('shortcut_candidate', 'feminine-coded'),
    'certainty':           ('shortcut_candidate', 'masculine-coded'),
    'negative_emotion':    ('shortcut_candidate', 'depression + feminine'),
    'swear_words':         ('shortcut_candidate', 'masculine-coded'),
    'excl_intensifiers':   ('shortcut_candidate', 'feminine-coded'),
    'question_density':    ('shortcut_candidate', 'feminine-coded'),
    'post_length':         ('shortcut_candidate', 'feminine-coded'),
    'apology_selfblame':   ('shortcut_candidate', 'feminine-coded'),
    'anger':               ('shortcut_candidate', 'masculine-coded'),
    'body_health':         ('clinical_reference', 'clinical + gendered'),
    'pain':                ('clinical_reference', 'clinical + gendered'),
    'cogproc':             ('gendered_reference', 'reference (fem + clinical)'),
    'self_ref_rumination': ('gendered_reference', 'reference (fem + clinical)'),
    'analytic':            ('gendered_reference', 'reference (gendered)'),
    'clout':               ('gendered_reference', 'reference (gendered)'),
}
FEATURE_NAMES = list(FEATURE_INFO.keys())
REFERENCE_FEATURES = {f for f, (tier, _) in FEATURE_INFO.items() if tier != 'shortcut_candidate'}


def _tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _count_sentences(text):
    sents = re.split(r'[.!?]+', text.strip())
    return max(len([s for s in sents if s.strip()]), 1)


def _row_features(row):
    text = str(row['text'])
    lower = text.lower()
    tokens = _tokenize(text)
    n_tok = max(len(tokens), 1)
    n_sent = _count_sentences(text)
    get = lambda col: row.get(col, 0) or 0  # LIWC column, defaults to 0 if absent/NaN

    hedge_count = sum(t in HYLAND_HEDGES_SINGLE for t in tokens)
    hedge_count += sum(lower.count(p) for p in HYLAND_HEDGES_MULTI)
    fp_sing = sum(t in FP_SINGULAR for t in tokens) / n_tok
    negative_emotion = get('emo_neg')
    empath_scores = _EMPATH.analyze(text, normalize=True) or {}

    return pd.Series({
        'hedge_density':       hedge_count / n_tok,
        'fp_singular':         fp_sing,
        'fp_plural':           sum(t in FP_PLURAL for t in tokens) / n_tok,
        'social_relational':   np.nanmean([get('Social'), get('family'), get('friend')]),
        'certainty':           get('certitude'),
        'negative_emotion':    negative_emotion,
        'swear_words':         get('swear'),
        'excl_intensifiers':   (text.count('!') + sum(t in INTENSIFIERS for t in tokens)) / n_tok,
        'question_density':    text.count('?') / n_sent,
        'post_length':         get('WC') or n_tok,
        'apology_selfblame':   sum(lower.count(p) for p in APOLOGY_PHRASES) / n_tok,
        'anger':               get('emo_anger'),
        'body_health':         np.nanmean([get('health'), get('Physical')]),
        'pain':                empath_scores.get('pain', 0.0),
        'cogproc':             get('cogproc'),
        'self_ref_rumination': fp_sing * negative_emotion,
        'analytic':            get('Analytic'),
        'clout':               get('Clout'),
    })


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 18 features as new columns. Requires a 'text' column; LIWC
    columns (Social, emo_neg, ...) are used if present, else default to 0.
    Some feature names (e.g. 'cogproc') intentionally equal their source LIWC
    column name -- drop the raw column first so concat doesn't duplicate it."""
    feats = df.apply(_row_features, axis=1)
    df = df.drop(columns=[c for c in feats.columns if c in df.columns])
    return pd.concat([df, feats.astype(float)], axis=1)
