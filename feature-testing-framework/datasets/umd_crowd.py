"""Loader for the UMD Reddit Suicidality Dataset (Shing et al. 2018 / Zirikly et
al. 2019), crowdsourced split (crowd/, not expert/) -- 1242 users (621
SuicideWatch-annotated a/b/c/d + 621 control), same risk scale and no LIWC
annotation as the expert split (see umd.py). Larger than expert (71k posts,
1242 users vs. 245/490), giving somewhat better control-group self-disclosure
coverage, though still thin (~16 gendered control users).

Build steps (see umd_crowd_prepare.py for the one-off raw-file assembly from
the tar.gz's crowd/train and crowd/test dirs):

    python extract_gender.py --input data/raw/umd_crowd/umd-crowd-raw.csv \
        --out-dir data/gender --user-col user_id --text-col text
    python run.py --dataset umd_crowd --data data/gender/umd-crowd-raw.csv
"""
import pandas as pd

from .umd import LABEL_NAMES


def load(path='data/gender/umd-crowd-raw.csv'):
    """Returns a DataFrame with columns: text, label (risk level name,
    'control' for users with no risk annotation), gender ('male'/'female')."""
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('')
    df['label'] = df['label'].map(LABEL_NAMES).fillna('control')
    return df


def conditions(df):
    return [c for c in LABEL_NAMES.values() if c in df['label'].unique()]
