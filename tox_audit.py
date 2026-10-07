"""
=============================================================
Real-Time Toxic Content Moderation
Optional: Hand-Audit -- how accurate is the model on Bluesky, really?
=============================================================
Author : Setu Patel

The model was trained on Wikipedia comments; Bluesky is different. The
flag rate on the dashboard therefore says what the model FLAGS, not how
much of Bluesky is toxic. This tool measures the difference: you review a
random sample of flagged replies by hand, and it computes how many were
genuinely abusive, with honest confidence intervals.

How to use it
-------------
1. Collect a sample while the live collector runs:
       python .\\tox_phase3_stream.py --audit 100
   (100 is a good size; the interval on a 100-comment estimate is about
   +/-10 points.) Let it run until the status line shows "audit 100/100".
2. Stop the collector (Ctrl+C), then review:
       python .\\tox_audit.py
   You can quit any time with q -- every answer is saved immediately, so
   you can review in several short sessions.
3. At the end it writes audit_summary.json (counts and rates only, no
   text), which the dashboard picks up automatically, and offers to delete
   the stored comment text.

Things to know before you start
-------------------------------
* CONTENT WARNING: you will read real flagged replies, and some will be
  offensive. Stop whenever you like; take breaks (it reminds you every 25).
* The sample's text lives ONLY in tox_audit.db, a separate local file. The
  main monitoring database and the dashboard never contain post text. At
  most N comments are stored (reservoir sampling); account names are never
  kept; comments deleted on Bluesky are removed from the sample too.
* The review is BLIND: the model's scores and categories are never shown,
  so they can't sway your judgment, and the order is shuffled.
* This measures PRECISION only (of what the model flags, how much is really
  bad). It does not measure what the model misses; that would need a much
  larger look at unflagged comments. You review one reply at a time without
  the post it replied to, which makes some calls ambiguous: use s to skip.

Your four answers
-----------------
  a  abusive      insults, harasses, threatens, demeans or attacks a person or group
  p  profane only crude or swearing language that is not aimed at attacking anyone
                  (for example enthusiastic swearing about a game or a meal)
  n  not toxic    neither of the above -- the model made a mistake
  s  skip         can't tell / too ambiguous / you'd rather not read it
  u  undo the last answer     q  quit and save
=============================================================
"""

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import textwrap
from datetime import datetime, timezone

from tox_common import LABEL_COLS

DEFAULT_AUDIT_DB = "tox_audit.db"
SUMMARY_FILE = "audit_summary.json"
ANSWERS = {"a": "abusive", "p": "profane only", "n": "not toxic", "s": "skip"}
BREAK_EVERY = 25
MIN_SAMPLE = 10

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS audit_samples (
    slot        INTEGER PRIMARY KEY,
    post_key    TEXT NOT NULL UNIQUE,
    scored_at   TEXT NOT NULL,
    text        TEXT,
    {', '.join(f'{c} REAL' for c in LABEL_COLS)},
    flags_f1    INTEGER NOT NULL,
    flags_hp    INTEGER NOT NULL,
    label       TEXT,
    reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS audit_meta (key TEXT PRIMARY KEY, value TEXT);
"""


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def init_audit_db(path):
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA secure_delete = ON")     # freed pages are overwritten, not just unlinked
    conn.executescript(SCHEMA)
    return conn


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM audit_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO audit_meta (key, value) VALUES (?, ?)", (key, str(value)))


# ── Sampling (used by the live collector) ──────────────────
class AuditReservoir:
    """Keeps a uniform random sample of at most `size` flagged comments from
    an endless stream (reservoir sampling, 'Algorithm R'): the i-th item seen
    replaces a random slot with probability size/i, so every flagged comment
    seen has the same chance of being in the sample, and storage never grows."""

    def __init__(self, path, size, seed=None, reset=False):
        if size < MIN_SAMPLE:
            raise SystemExit(f"--audit needs at least {MIN_SAMPLE} (a sample smaller than that tells you little).")
        if reset:
            for suffix in ("", "-journal", "-wal", "-shm"):
                try:
                    os.remove(path + suffix)
                except OSError:
                    pass
        self.conn = init_audit_db(path)
        self.size = size
        self.rng = random.Random(seed)
        reviewed = self.conn.execute("SELECT COUNT(*) FROM audit_samples WHERE label IS NOT NULL").fetchone()[0]
        if reviewed:
            self.conn.close()
            raise SystemExit(
                f"{path} already holds {reviewed} reviewed comment(s). Adding to it would disturb your results.\n"
                "Finish up with:  python .\\tox_audit.py --summary\n"
                "or start a fresh sample with:  --audit-reset")
        stored_size = get_meta(self.conn, "size")
        if stored_size is not None and int(stored_size) != size and self.count_rows() > 0:
            self.conn.close()
            raise SystemExit(f"{path} holds a sample collected with --audit {stored_size}. Use the same size to "
                             "continue it, or --audit-reset to start over.")
        set_meta(self.conn, "size", size)
        self.seen = int(get_meta(self.conn, "seen_flagged", 0))
        self.slot_of = {k: s for s, k in self.conn.execute("SELECT slot, post_key FROM audit_samples")}
        self.occupied = set(self.slot_of.values())
        self.conn.commit()

    def count_rows(self):
        return self.conn.execute("SELECT COUNT(*) FROM audit_samples").fetchone()[0]

    def count(self):
        return len(self.slot_of)

    def _free_slot(self):
        for s in range(1, self.size + 1):
            if s not in self.occupied:
                return s
        return None

    def offer(self, key, scored_at, text, scores, flags_f1, flags_hp):
        """Considers one flagged comment. Returns True if it was kept."""
        if key in self.slot_of:
            return False
        self.seen += 1
        slot = self._free_slot()
        if slot is None:
            j = self.rng.randint(1, self.seen)
            if j > self.size:
                set_meta(self.conn, "seen_flagged", self.seen)
                self.conn.commit()
                return False
            slot = j
        old = self.conn.execute("SELECT post_key FROM audit_samples WHERE slot = ?", (slot,)).fetchone()
        self.conn.execute(
            f"INSERT OR REPLACE INTO audit_samples (slot, post_key, scored_at, text, {', '.join(LABEL_COLS)}, "
            f"flags_f1, flags_hp) VALUES (?, ?, ?, ?, {', '.join('?' * len(LABEL_COLS))}, ?, ?)",
            (slot, key, scored_at, text, *[float(s) for s in scores], int(flags_f1), int(flags_hp)))
        set_meta(self.conn, "seen_flagged", self.seen)
        self.conn.commit()
        if old:
            self.slot_of.pop(old[0], None)
        self.slot_of[key] = slot
        self.occupied.add(slot)
        return True

    def remove(self, keys):
        """The author deleted the post on Bluesky -> delete it from the sample too."""
        hit = [k for k in keys if k in self.slot_of]
        for k in hit:
            self.conn.execute("DELETE FROM audit_samples WHERE post_key = ?", (k,))
            self.occupied.discard(self.slot_of.pop(k))
        if hit:
            self.conn.commit()
        return len(hit)

    def close(self):
        self.conn.close()


# ── Statistics ─────────────────────────────────────────────
def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n. Preferred over the
    textbook +/- formula, which is unreliable for small samples or rates
    near 0% or 100%."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _block(labels):
    n = len(labels)
    a, p, nn = labels.count("a"), labels.count("p"), labels.count("n")
    if n == 0:
        return {"n": 0, "abusive": 0, "profane_only": 0, "not_toxic": 0,
                "abusive_rate": None, "abusive_ci": None, "offensive_rate": None, "offensive_ci": None}
    return {"n": n, "abusive": a, "profane_only": p, "not_toxic": nn,
            "abusive_rate": a / n, "abusive_ci": list(wilson_interval(a, n)),
            "offensive_rate": (a + p) / n, "offensive_ci": list(wilson_interval(a + p, n))}


def summarize(conn):
    rows = conn.execute("SELECT label, flags_hp FROM audit_samples").fetchall()
    reviewed = [(l, hp) for l, hp in rows if l in ("a", "p", "n")]
    return {
        "created_at": utc_now_iso(),
        "population": "replies flagged in Sensitive mode while audit mode was on; Strict-mode figures use the "
                      "subset that was also flagged in Strict mode",
        "sample_size": len(rows),
        "n_reviewed": len(reviewed),
        "n_skipped": sum(1 for l, _ in rows if l == "s"),
        "n_unreviewed": sum(1 for l, _ in rows if l is None),
        "modes": {"sensitive": _block([l for l, _ in reviewed]),
                  "strict": _block([l for l, hp in reviewed if hp != 0])},
        "caveats": ["one reviewer", "blind to model scores", "no parent post shown for context",
                    "precision only; says nothing about what the model misses"],
    }


def print_summary(s, say=print):
    say(f"\nResults: {s['n_reviewed']} reviewed, {s['n_skipped']} skipped, {s['n_unreviewed']} not yet reviewed "
        f"(sample of {s['sample_size']})")
    for key, name in (("strict", "Strict mode"), ("sensitive", "Sensitive mode")):
        b = s["modes"][key]
        if not b["n"]:
            say(f"  {name}: no reviewed comments yet")
            continue
        lo, hi = b["abusive_ci"]
        say(f"  {name} ({b['n']} reviewed): {b['abusive_rate']:.0%} abusive (95% CI {lo:.0%}-{hi:.0%}), "
            f"{b['profane_only'] / b['n']:.0%} profane only, {b['not_toxic'] / b['n']:.0%} not toxic")
    say("  Precision only. One reviewer, blind to the model, without the parent post for context.")


def write_summary(s, path=SUMMARY_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)


def purge_text(conn):
    """Deletes every stored comment text, keeping labels and scores. VACUUM
    rewrites the file so the text is not left behind in free pages."""
    conn.execute("UPDATE audit_samples SET text = NULL")
    conn.commit()
    conn.execute("VACUUM")


# ── Review ─────────────────────────────────────────────────
INTRO = f"""
HAND-AUDIT
==========
CONTENT WARNING: you are about to read real replies that the model flagged.
Some will contain offensive language. You can stop at any time (q); every
answer is saved the moment you give it.

For each reply, answer with one key:
  a  abusive      insults, harasses, threatens, demeans or attacks someone
  p  profane only crude/swearing language NOT aimed at attacking anyone
  n  not toxic    neither -- the model made a mistake
  s  skip         can't tell, too ambiguous, or you'd rather not read it
  u  undo your last answer          q  quit and save

You will not see the model's scores. You will not see the post being replied
to, so skip anything you can't judge without it.
"""

LEGEND = "  [a] abusive   [p] profane only   [n] not toxic   [s] skip   [u] undo   [q] quit"


def review_order(conn):
    seed = get_meta(conn, "review_seed")
    if seed is None:
        seed = random.SystemRandom().randint(1, 10 ** 9)
        set_meta(conn, "review_seed", seed)
        conn.commit()
    slots = [r[0] for r in conn.execute("SELECT slot FROM audit_samples ORDER BY slot")]
    random.Random(int(seed)).shuffle(slots)
    return slots


def review_loop(conn, input_fn=input, say=print, break_every=BREAK_EVERY):
    order = review_order(conn)
    todo = [s for s in order
            if conn.execute("SELECT label, text FROM audit_samples WHERE slot = ?", (s,)).fetchone()[0] is None]
    total = len(order)
    i, session_done, history = 0, 0, []
    while i < len(todo):
        slot = todo[i]
        text = conn.execute("SELECT text FROM audit_samples WHERE slot = ?", (slot,)).fetchone()[0]
        if text is None:                      # text already purged: nothing to review
            i += 1
            continue
        done = conn.execute("SELECT COUNT(*) FROM audit_samples WHERE label IS NOT NULL").fetchone()[0]
        say("\n" + "-" * 78)
        say(f"Reply {done + 1} of {total}")
        say(textwrap.fill(text, width=76, initial_indent="  ", subsequent_indent="  "))
        say(LEGEND)
        while True:
            try:
                ans = input_fn("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "q"
            if ans in ANSWERS:
                conn.execute("UPDATE audit_samples SET label = ?, reviewed_at = ? WHERE slot = ?",
                             (ans, utc_now_iso(), slot))
                conn.commit()
                history.append(slot)
                i += 1
                session_done += 1
                break
            if ans == "u":
                if not history:
                    say("  Nothing to undo yet.")
                    continue
                prev = history.pop()
                conn.execute("UPDATE audit_samples SET label = NULL, reviewed_at = NULL WHERE slot = ?", (prev,))
                conn.commit()
                i -= 1
                session_done -= 1          # an undone answer no longer counts toward the session total
                break
            if ans == "q":
                return session_done
            say("  Please type a, p, n, s, u or q.")
        if break_every and session_done and session_done % break_every == 0 and i < len(todo) and ans in ANSWERS:
            say(f"\nThat's {session_done} this session. Take a break if you like.")
            try:
                if input_fn("Press Enter to continue, or q to stop: ").strip().lower() == "q":
                    return session_done
            except (EOFError, KeyboardInterrupt):
                return session_done
    return session_done


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hand-review a sample of flagged comments and measure precision.")
    p.add_argument("--db", default=DEFAULT_AUDIT_DB)
    p.add_argument("--out", default=SUMMARY_FILE)
    p.add_argument("--summary", action="store_true", help="just compute and print results (no review)")
    p.add_argument("--purge", action="store_true", help="delete the stored comment text and exit")
    return p.parse_args(argv)


def main(argv=None, input_fn=input, say=print):
    args = parse_args(argv)
    if not os.path.exists(args.db):
        say(f"No audit sample found ({args.db}). Collect one first:  python .\\tox_phase3_stream.py --audit 100")
        return 1
    conn = init_audit_db(args.db)
    try:
        if args.purge:
            purge_text(conn)
            say("Stored comment text deleted. Labels and scores kept.")
            return 0
        if not args.summary:
            n = conn.execute("SELECT COUNT(*) FROM audit_samples").fetchone()[0]
            left = conn.execute("SELECT COUNT(*) FROM audit_samples WHERE label IS NULL AND text IS NOT NULL").fetchone()[0]
            if n == 0:
                say("The sample is empty. Let the collector run longer with --audit.")
                return 1
            say(INTRO)
            say(f"Sample: {n} comments, {left} still to review.")
            try:
                if input_fn("Press Enter to begin, or q to quit: ").strip().lower() == "q":
                    return 0
            except (EOFError, KeyboardInterrupt):
                return 0
            review_loop(conn, input_fn, say)
        s = summarize(conn)
        print_summary(s, say)
        if s["n_reviewed"]:
            write_summary(s, args.out)
            say(f"\nWrote {args.out} (counts and rates only). The dashboard will show it.")
        if not args.summary and s["n_unreviewed"] == 0:
            try:
                ans = input_fn("\nDelete the stored comment text now? Your labels and the scores are kept. [y/N]: ")
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            if ans.strip().lower() == "y":
                purge_text(conn)
                say("Stored comment text deleted.")
            else:
                say("Text kept. Delete it later with:  python .\\tox_audit.py --purge")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")    # emoji etc. must never crash the review on Windows
    except AttributeError:
        pass
    sys.exit(main())
