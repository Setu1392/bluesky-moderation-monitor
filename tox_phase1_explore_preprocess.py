"""
=============================================================
Real-Time Toxic Content Moderation
Phase 1: Data Exploration & Preprocessing
=============================================================
Author : Setu Patel
Data   : Jigsaw Toxic Comment Classification Challenge
         ~160K Wikipedia talk-page comments, each labeled for six
         kinds of toxicity (multi-label: one comment can carry
         several flags at once). Public mirror on Hugging Face --
         labels are CC0, comment text is CC-BY-SA 3.0.
Output : toxic_comments.db               (SQLite: raw comments + derived fields)
         processed/train.csv, val.csv, test.csv
         processed/preprocessing_config.json

Design notes
------------
* No authentication needed. The original Kaggle competition download
  requires accepting the competition rules in a browser first, and
  Kaggle API tokens expire after a few hours; this mirror avoids both.
* This script deliberately NEVER prints raw comment text. The dataset
  contains genuinely offensive language, and the live dashboard (Phase 4)
  masks flagged content by default -- the exploration tooling follows
  the same rule, so nothing offensive lands in your terminal or in a
  screenshot of it.
* Text cleaning lives in tox_common.py and is shared by every phase, so
  the identical function runs on training data here and on live Reddit
  comments later (see that file for why this matters).
* Run order for the whole project:
    tox_phase1_explore_preprocess.py   (this file)
=============================================================
"""

import os
import re
import json
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from tox_common import LABEL_COLS, clean_text

# ── Config ─────────────────────────────────────────────────
TRAIN_URL = (
    "https://huggingface.co/datasets/thesofakillers/"
    "jigsaw-toxic-comment-classification-challenge/resolve/main/train.csv?download=true"
)
RAW_DIR = os.path.join("data", "raw")
RAW_PATH = os.path.join(RAW_DIR, "train.csv")
DB_PATH = "toxic_comments.db"
OUT_DIR = "processed"
SEED = 42

# Candidate vocabulary sizes to evaluate; the smallest one that covers
# at least VOCAB_COVERAGE_TARGET of all word occurrences gets recommended.
VOCAB_CANDIDATES = [10_000, 20_000, 30_000, 50_000]
VOCAB_COVERAGE_TARGET = 0.97
MAX_LEN_CAP = 400  # never recommend a sequence length longer than this

TOKEN_RE = re.compile(r"[a-z0-9']+")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
pd.set_option("display.width", 120)


# ── Step 1: Get the data ───────────────────────────────────
def download_file(url, dest):
    """Streams `url` to `dest`. Writes to a .part file first and only
    renames on success, so an interrupted download can never be mistaken
    for a complete file on the next run."""
    print(f"  [download] {url.split('?')[0]}")
    part_path = dest + ".part"
    try:
        resp = requests.get(
            url,
            stream=True,
            timeout=(10, 60),
            headers={"User-Agent": "tox-moderation-portfolio/1.0"},
        )
    except requests.RequestException as e:
        raise SystemExit(manual_download_help(f"network error: {e}"))

    if resp.status_code != 200:
        raise SystemExit(manual_download_help(f"HTTP {resp.status_code}"))

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(part_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r    {downloaded / 1e6:.0f} MB / {total / 1e6:.0f} MB "
                      f"({downloaded / total * 100:.0f}%)", end="")
            else:
                print(f"\r    {downloaded / 1e6:.0f} MB", end="")
    print()
    os.replace(part_path, dest)
    print(f"    -> saved to {dest}")


def manual_download_help(reason):
    return (
        f"\nDownload failed ({reason}).\n"
        "Manual fallback: open this link in your browser, save the file as\n"
        f"    {RAW_PATH}\n"
        "and re-run this script (it will skip the download step):\n"
        "    https://huggingface.co/datasets/thesofakillers/"
        "jigsaw-toxic-comment-classification-challenge/blob/main/train.csv"
    )


def ensure_raw_data():
    if os.path.exists(RAW_PATH):
        print(f"  [skip] {RAW_PATH} already present")
    else:
        download_file(TRAIN_URL, RAW_PATH)


# ── Step 2: Load and validate ──────────────────────────────
def load_raw():
    df = pd.read_csv(RAW_PATH)

    expected = ["id", "comment_text"] + LABEL_COLS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{RAW_PATH} is missing expected columns {missing}. If this file "
            "came from a failed or interrupted download (e.g. an HTML error page "
            "saved as .csv), delete it and re-run."
        )

    n_null_text = int(df["comment_text"].isna().sum())
    bad_labels = int((~df[LABEL_COLS].isin([0, 1])).any(axis=1).sum())
    n_dup_ids = int(df["id"].duplicated().sum())
    n_dup_text = int(df["comment_text"].duplicated().sum())

    print(f"  Rows loaded            : {len(df):,}")
    print(f"  Null comment_text      : {n_null_text}")
    print(f"  Rows with non-0/1 label: {bad_labels}")
    print(f"  Duplicate ids          : {n_dup_ids}")
    print(f"  Duplicate comment text : {n_dup_text:,} "
          "(removed before splitting, to prevent train/test leakage)")

    if n_null_text or bad_labels:
        raise SystemExit("Unexpected nulls or label values -- inspect the raw file.")
    return df


# ── Step 3: Derived fields used for exploration ────────────
def add_derived_fields(df):
    df = df.copy()
    df["n_labels"] = df[LABEL_COLS].sum(axis=1)
    df["any_toxic"] = (df["n_labels"] > 0).astype(int)
    df["char_len"] = df["comment_text"].str.len()

    # Fraction of letters that are uppercase -- a cheap "shouting" signal.
    upper = df["comment_text"].str.count(r"[A-Z]")
    letters = df["comment_text"].str.count(r"[A-Za-z]").clip(lower=1)
    df["upper_ratio"] = (upper / letters).round(4)

    # Simple regex tokenization, used ONLY for these exploration stats.
    # Keras' TextVectorization (Phase 2) tokenizes slightly differently,
    # so treat the counts here as close estimates, not exact figures.
    counter = Counter()
    word_counts = []
    for text in df["comment_text"].str.lower():
        tokens = TOKEN_RE.findall(text)
        word_counts.append(len(tokens))
        counter.update(tokens)
    df["word_count"] = word_counts
    return df, counter


# ── Step 4: SQLite + SQL exploration ───────────────────────
def write_to_sqlite(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("comments", conn, if_exists="replace", index=False, chunksize=20_000)
    with conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_any_toxic ON comments(any_toxic)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_n_labels ON comments(n_labels)")
    conn.close()
    print(f"  [db] comments -> {len(df):,} rows saved to {DB_PATH}")


def run_query(conn, sql, title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)
    out = pd.read_sql(sql, conn)
    print(out.to_string(index=False))
    return out


def sql_exploration():
    conn = sqlite3.connect(DB_PATH)

    parts = [
        f"SELECT '{c}' AS label, SUM({c}) AS positives, "
        f"ROUND(100.0 * SUM({c}) / COUNT(*), 3) AS pct_of_comments FROM comments"
        for c in LABEL_COLS
    ]
    parts.append(
        "SELECT 'ANY label' AS label, SUM(any_toxic) AS positives, "
        "ROUND(100.0 * SUM(any_toxic) / COUNT(*), 3) AS pct_of_comments FROM comments"
    )
    run_query(
        conn,
        "SELECT * FROM (" + " UNION ALL ".join(parts) + ") ORDER BY positives DESC",
        "1. Class balance — how rare is each kind of toxicity?",
    )

    run_query(conn, """
        SELECT n_labels AS labels_on_one_comment,
               COUNT(*) AS comments,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM comments), 2) AS pct
        FROM comments
        GROUP BY n_labels
        ORDER BY n_labels
    """, "2. How many flags does a single comment carry?")

    run_query(conn, """
        SELECT CASE any_toxic WHEN 1 THEN 'flagged (any label)' ELSE 'clean' END AS group_name,
               COUNT(*)                         AS comments,
               ROUND(AVG(char_len), 1)          AS avg_chars,
               ROUND(AVG(word_count), 1)        AS avg_words,
               ROUND(AVG(upper_ratio), 3)       AS avg_upper_ratio
        FROM comments
        GROUP BY any_toxic
    """, "3. Do flagged comments look different from clean ones?")

    run_query(conn, """
        SELECT CASE
                   WHEN word_count < 10  THEN '1) under 10 words'
                   WHEN word_count < 30  THEN '2) 10-29 words'
                   WHEN word_count < 100 THEN '3) 30-99 words'
                   WHEN word_count < 250 THEN '4) 100-249 words'
                   ELSE                       '5) 250+ words'
               END AS length_bucket,
               COUNT(*) AS comments,
               ROUND(100.0 * SUM(any_toxic) / COUNT(*), 2) AS flagged_rate_pct
        FROM comments
        GROUP BY length_bucket
        ORDER BY length_bucket
    """, "4. Flagged rate by comment length")

    run_query(conn, """
        SELECT CASE
                   WHEN upper_ratio >= 0.5 THEN '3) mostly CAPS (50%+ of letters)'
                   WHEN upper_ratio >= 0.2 THEN '2) some CAPS (20-49%)'
                   ELSE                         '1) normal case (under 20%)'
               END AS shouting_bucket,
               COUNT(*) AS comments,
               ROUND(100.0 * SUM(any_toxic) / COUNT(*), 2) AS flagged_rate_pct
        FROM comments
        WHERE word_count >= 5
        GROUP BY shouting_bucket
        ORDER BY shouting_bucket
    """, "5. Flagged rate by 'shouting' (comments of 5+ words)")

    conn.close()


# ── Step 5: Pandas-side exploration ────────────────────────
def cooccurrence_report(df):
    print(f"\n{'─' * 60}")
    print("  6. Label co-occurrence — given a comment has the ROW label,")
    print("     what % of the time does it ALSO have the COLUMN label?")
    print("─" * 60)
    cooc = df[LABEL_COLS].T.dot(df[LABEL_COLS])
    cond = cooc.div(np.diag(cooc), axis=0) * 100
    print(cond.round(1).to_string())


def length_and_vocab_report(df, counter):
    print(f"\n{'─' * 60}")
    print("  7. Sequence length — sizing the model's input window")
    print("─" * 60)
    quantiles = df["word_count"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    for q, v in quantiles.items():
        print(f"    {int(q * 100):>3}th percentile: {int(v):>5} words")
    p95 = int(quantiles.loc[0.95])
    max_len = int(min(MAX_LEN_CAP, int(np.ceil(p95 / 10.0)) * 10))
    print(f"    -> recommended max sequence length: {max_len} "
          f"(95th percentile, rounded up, capped at {MAX_LEN_CAP})")

    print(f"\n{'─' * 60}")
    print("  8. Vocabulary — how many distinct words are worth keeping?")
    print("─" * 60)
    freqs = np.array(sorted(counter.values(), reverse=True))
    total = freqs.sum()
    cum = np.cumsum(freqs)
    print(f"    Distinct words seen: {len(freqs):,}   Total word occurrences: {int(total):,}")
    recommended = VOCAB_CANDIDATES[-1]
    found = False
    for k in VOCAB_CANDIDATES:
        coverage = cum[min(k, len(cum)) - 1] / total
        print(f"    Top {k:>6,} words cover {coverage * 100:5.2f}% of all occurrences")
        if not found and coverage >= VOCAB_COVERAGE_TARGET:
            recommended, found = k, True
    print(f"    -> recommended vocabulary size: {recommended:,} "
          f"(smallest tested size covering >= {VOCAB_COVERAGE_TARGET * 100:.0f}% of occurrences)")

    return {"recommended_max_len": max_len, "recommended_vocab_size": int(recommended)}


# ── Step 6: Clean, split, save ─────────────────────────────
def build_splits(df):
    before = len(df)
    df = df.drop_duplicates(subset="comment_text", keep="first").copy()
    print(f"  Removed {before - len(df):,} exact-duplicate comments "
          f"({len(df):,} remain)")

    df["text"] = df["comment_text"].map(clean_text)

    # Stratify on the RAREST label a comment carries, not just "toxic or
    # not". With plain any_toxic stratification, a rare label like
    # `threat` (a fraction of a percent of rows) can land almost entirely
    # in one split by chance, making its per-label metrics meaningless.
    df["strat_key"] = np.select(
        [df["threat"] == 1, df["identity_hate"] == 1,
         df["severe_toxic"] == 1, df["any_toxic"] == 1],
        ["threat", "identity_hate", "severe_toxic", "toxic_other"],
        default="clean",
    )
    smallest = df["strat_key"].value_counts().min()
    if smallest < 10:
        print(f"  [warn] smallest stratum has only {smallest} rows; "
              "falling back to stratifying on any_toxic")
        df["strat_key"] = df["any_toxic"].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df["strat_key"], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["strat_key"], random_state=SEED)

    return train_df, val_df, test_df


def split_quality_report(train_df, val_df, test_df):
    print(f"\n{'─' * 60}")
    print("  9. Split quality — positive rate (%) per label in each split")
    print("     (rates should look nearly identical across rows)")
    print("─" * 60)
    rows = []
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        row = {"split": name, "n": len(part)}
        for c in LABEL_COLS:
            row[c] = round(part[c].mean() * 100, 3)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  Positives per label in the TEST split "
          "(small counts = noisy per-label metrics later):")
    print("    " + ", ".join(f"{c}: {int(test_df[c].sum())}" for c in LABEL_COLS))


def save_outputs(train_df, val_df, test_df, rec, n_dupes_removed):
    cols = ["id", "text"] + LABEL_COLS
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        part[cols].to_csv(path, index=False)
        print(f"  [saved] {path}  ({len(part):,} rows)")

    config = {
        "label_cols": LABEL_COLS,
        "recommended_max_len": rec["recommended_max_len"],
        "recommended_vocab_size": rec["recommended_vocab_size"],
        "random_seed": SEED,
        "stratification": "rarest label present (threat > identity_hate > severe_toxic > other toxic > clean)",
        "duplicates_removed_before_split": int(n_dupes_removed),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "train_positive_rate": {c: float(train_df[c].mean()) for c in LABEL_COLS},
    }
    path = os.path.join(OUT_DIR, "preprocessing_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  [saved] {path}")


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Toxic Moderation — Phase 1: Exploration & Preprocessing")
    print("=" * 60)

    print("\n[1/6] Getting the data ...")
    ensure_raw_data()

    print("\n[2/6] Loading and validating ...")
    raw = load_raw()

    print("\n[3/6] Deriving exploration fields ...")
    df, counter = add_derived_fields(raw)

    print("\n[4/6] Loading into SQLite and running SQL exploration ...")
    write_to_sqlite(df)
    sql_exploration()

    print("\n[5/6] Pandas-side exploration ...")
    cooccurrence_report(df)
    rec = length_and_vocab_report(df, counter)

    print("\n[6/6] Cleaning, splitting, saving ...")
    n_before = len(df)
    train_df, val_df, test_df = build_splits(df)
    n_dupes_removed = n_before - (len(train_df) + len(val_df) + len(test_df))
    split_quality_report(train_df, val_df, test_df)
    save_outputs(train_df, val_df, test_df, rec, n_dupes_removed)

    print("""
          ── Phase 1 output, ready for Phase 2 ──────────────────────
          toxic_comments.db                 raw comments + derived fields (SQL-queryable)
          processed/train.csv               80% of data, cleaned text + 6 labels
          processed/val.csv                 10%, for tuning during training
          processed/test.csv                10%, touched ONCE at the end for final metrics
          processed/preprocessing_config.json   recommended max length + vocab size

          Phase 2 will train a TensorFlow multi-label classifier on these files.""")
