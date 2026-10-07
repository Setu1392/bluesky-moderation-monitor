"""
=============================================================
Real-Time Toxic Content Moderation
Phase 3: Live Stream Scoring (Bluesky Jetstream) + SQLite Log
=============================================================
Author : Setu Patel
Requires: models/ from tox_phase2_train.py, and:  pip install websockets
Output : tox_stream.db   (read by the Phase 4 dashboard)

Three ways to run it, safest first:

  python .\\tox_phase3_stream.py --probe
      15-second NETWORK check. Connects to Bluesky, counts what arrives and
      prints the *structure* (key names only, never values or post text) of
      the first messages. Needs no model and stores nothing. Run this first.

  python .\\tox_phase3_stream.py --selftest
      OFFLINE end-to-end test: synthetic events -> your real trained model ->
      SQLite, checking deduplication, deletion handling and that no post
      text is stored. Needs no network.

  python .\\tox_phase3_stream.py
      The real thing, until Ctrl+C. (--duration-min N stops it after N minutes.)

  python .\\tox_phase3_stream.py --audit 100
      Same, but ALSO keeps the text of up to 100 randomly chosen flagged replies
      in a separate local file (tox_audit.db) so you can hand-check them with
      tox_audit.py and measure how accurate the model really is on Bluesky.
      Opt-in, local, capped at N comments, and purgeable -- see tox_audit.py.

Where the data comes from
-------------------------
Bluesky's public Jetstream service (v2) streams every public post as JSON
over one WebSocket, free, with no account or key. This script asks for
posts only, keeps English replies (Bluesky's version of comments), scores
each with the Phase 2 model, and logs the results.

Privacy and responsibility (by design, not by accident)
-------------------------------------------------------
* The post text is NEVER stored in the monitoring database or printed (the one
  exception is the opt-in --audit sample above, kept in its own file). It keeps the six
  scores, text length/shape, a hashed post key, and a fully masked preview
  (every letter replaced by a bullet). No account names or IDs are kept.
* If a user deletes a post, the matching database row is deleted too, and a
  short-lived "tombstone" memory stops a deleted post from being stored if
  its delete event arrives before it has been scored.
* Read-only: this never posts, replies, likes, or notifies anyone.
* Raw rows older than --keep-hours (default 48) are pruned automatically.

Threshold bitmasks (for Phase 4): `flags_f1` and `flags_hp` are integers
where bit j is set if label j (in this order: toxic, severe_toxic, obscene,
threat, insult, identity_hate) crossed its threshold in that mode.
=============================================================
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
import calendar
import hashlib
import json
import queue
import random
import re
import sqlite3
import string
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import numpy as np

from tox_audit import AuditReservoir, DEFAULT_AUDIT_DB
from tox_common import LABEL_COLS, clean_text, upper_ratio
from tox_scoring import load_artifacts, make_inputs, predict_probs, score_texts, threshold_vector

# ── Config ─────────────────────────────────────────────────
ENDPOINTS = [
    "wss://jetstream.us-east.bsky.network",
    "wss://jetstream.us-west.bsky.network",
]
SUBSCRIBE_PATH = "/xrpc/network.bsky.jetstream.subscribeEvents"
SUBPROTOCOL = "xrpc.v1.json"
COLLECTION = "app.bsky.feed.post"

QUEUE_MAX = 2000           # scoring backlog before the OLDEST posts get dropped
BATCH_SIZE = 32
BATCH_WINDOW_S = 2.0       # max wait to fill a batch (also the pipeline's typical lag)
RECV_TIMEOUT_S = 30.0      # no data for this long = dead connection, reconnect
STATS_INTERVAL_S = 10.0
PRUNE_INTERVAL_S = 600.0
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 30.0
MIN_CHARS = 5
MAX_CHARS = 1000
PREVIEW_CHARS = 48
TOMBSTONE_MAX = 200_000
RESUME_REWIND_US = 2_000_000   # on reconnect, resume 2 s before the last event seen
MAX_RESUME_AGE_S = 120         # older than this (laptop slept, long outage): rejoin live instead of replaying
MAX_RESUME_FAILURES = 3        # this many failed attempts WITH a resume point: give the resume point up
DEFAULT_DB = "tox_stream.db"
SELFTEST_DB = "tox_selftest.db"

_MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9][A-Za-z0-9._-]*")
_KEEP_VISIBLE = set(string.punctuation)
_TIME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})?$")


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Text helpers ───────────────────────────────────────────
def clean_social_text(text):
    """Bluesky-specific tidying, then the same shared cleaning used in training.
    @mentions are removed (the handle is noise to the model and an identity we
    don't want to keep) and '#' is dropped. The model never saw either in
    Wikipedia text, so this adapts the input rather than creating skew."""
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("#", "")
    return clean_text(text)


def mask_preview(text, n=PREVIEW_CHARS):
    """First n characters with every letter, digit and symbol (including
    emoji) replaced by a bullet. Whitespace and ASCII punctuation stay, so the
    preview shows the shape of a comment but none of its content."""
    snippet = text[:n]
    masked = "".join(ch if (ch.isspace() or ch in _KEEP_VISIBLE) else "\u2022" for ch in snippet)
    return masked + ("\u2026" if len(text) > n else "")


def post_key(did, rkey):
    """Stable pseudonymous key for a post, derived from its at:// URI. Used for
    deduplication and for honoring deletions; the URI itself is not kept."""
    uri = f"at://{did}/{COLLECTION}/{rkey}"
    return hashlib.sha256(uri.encode("utf-8")).hexdigest()[:24]


def pack_mask(flags):
    return int(sum(1 << j for j, f in enumerate(flags) if f))


# ── Event helpers (Jetstream v2 format) ────────────────────
def build_url(endpoint, cursor=None):
    params = [("collections", COLLECTION), ("kinds", "commit")]
    if cursor is not None:
        params.append(("cursor", str(cursor)))
    return f"{endpoint}{SUBSCRIBE_PATH}?{urlencode(params)}"


def parse_time_us(s):
    """RFC 3339 timestamp string -> unix microseconds (UTC), or None.
    Handles any number of fractional digits (truncated to microseconds)."""
    if not isinstance(s, str):
        return None
    m = _TIME_RE.match(s)
    if not m:
        return None
    y, mo, d, h, mi, sec = (int(m.group(i)) for i in range(1, 7))
    frac = (m.group(7) or "")[:6].ljust(6, "0")
    try:
        epoch = calendar.timegm((y, mo, d, h, mi, sec, 0, 0, 0))
    except (ValueError, OverflowError):
        return None
    tz, offset = m.group(8), 0
    if tz and tz != "Z":
        offset = (1 if tz[0] == "+" else -1) * (int(tz[1:3]) * 3600 + int(tz[4:6]) * 60)
    return (epoch - offset) * 1_000_000 + int(frac)


def decode_frame(raw):
    """One WebSocket frame -> (cursor_us, kind, key, record).

    v2 frames look like {"$type": "message", "payload": {...}} with the commit
    fields (did, time, operation, collection, rkey, record) inside `payload`.
    kind is "create" or "delete" for posts, else None (identity/account events,
    other collections, edits, malformed input). `cursor_us` is the event's own
    timestamp in unix microseconds, returned whenever it can be read so that a
    reconnect can resume from the right moment -- a timestamp cursor works on
    any Jetstream instance, unlike a sequence number whose numbering may be
    specific to one server."""
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return None, None, None, None
    if not isinstance(msg, dict):
        return None, None, None, None
    payload = msg.get("payload")
    if not isinstance(payload, dict):
        return None, None, None, None
    cursor_us = parse_time_us(payload.get("time"))
    if payload.get("collection") != COLLECTION or "operation" not in payload:
        return cursor_us, None, None, None
    did, rkey = payload.get("did"), payload.get("rkey")
    if not isinstance(did, str) or not isinstance(rkey, str):
        return cursor_us, None, None, None
    key = post_key(did, rkey)
    op = payload.get("operation")
    if op == "delete":
        return cursor_us, "delete", key, None
    if op == "create" and isinstance(payload.get("record"), dict):
        return cursor_us, "create", key, payload["record"]
    return cursor_us, None, None, None


def prepare_post(record, replies_only=True, lang="en"):
    """Filter and clean one post record. Returns a dict, or None to skip."""
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    is_reply = isinstance(record.get("reply"), dict)
    if replies_only and not is_reply:
        return None
    langs = record.get("langs")
    langs = [l for l in langs if isinstance(l, str)] if isinstance(langs, list) else []
    matched = None
    if lang:
        for l in langs:
            if l.lower().split("-")[0] == lang:
                matched = l
                break
        if matched is None:
            return None
    else:
        matched = langs[0] if langs else None
    cleaned = clean_social_text(text[:MAX_CHARS])
    if len(cleaned) < MIN_CHARS:
        return None
    created = record.get("createdAt")
    return {
        "text": cleaned,
        "lang": matched,
        "is_reply": int(is_reply),
        "created_at": created[:32] if isinstance(created, str) else None,
    }


# ── Shared state ───────────────────────────────────────────
class Stats:
    FIELDS = ("seen", "passed", "dropped", "scored", "flagged_f1", "flagged_hp",
              "deletes", "reconnects", "lag_sum")

    def __init__(self):
        self._lock = threading.Lock()
        self._v = {k: 0 for k in self.FIELDS}

    def add(self, **kw):
        with self._lock:
            for k, v in kw.items():
                self._v[k] += v

    def snapshot(self):
        with self._lock:
            return dict(self._v)


class StreamState:
    def __init__(self):
        self.cursor_us = None     # newest event timestamp seen (unix microseconds)
        self.connected = False
        self.endpoint = None
        self.error = None


class Tombstones:
    """Remembers recently deleted post keys (bounded), so a post whose delete
    event is processed before the post itself is never stored."""

    def __init__(self, maxlen=TOMBSTONE_MAX):
        self._order = deque()
        self._set = set()
        self._max = maxlen

    def add(self, key):
        if key in self._set:
            return
        self._set.add(key)
        self._order.append(key)
        if len(self._order) > self._max:
            self._set.discard(self._order.popleft())

    def __contains__(self, key):
        return key in self._set


# ── Frame sources ──────────────────────────────────────────
def sleep_interruptible(stop, seconds):
    end = time.monotonic() + seconds
    while not stop.is_set() and time.monotonic() < end:
        time.sleep(min(0.1, max(0.0, end - time.monotonic())))


def websocket_frames(endpoints, stop, state, stats, say=log, recv_timeout=RECV_TIMEOUT_S):
    """Yields raw frames from Jetstream forever, reconnecting with exponential
    backoff + jitter, rotating between endpoints after failures, and resuming
    shortly before the newest event seen (`state.cursor_us`, kept by the
    reader). Delivery is at-least-once across a reconnect; the database's
    unique post key makes the duplicates harmless."""
    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("The 'websockets' package is missing. Run:  pip install websockets")

    attempt = 0
    resume_failures = 0
    while not stop.is_set():
        endpoint = endpoints[attempt % len(endpoints)]
        # Decide whether to ask the server to replay what we missed. Replaying is only worth
        # it after a brief blip: after a long gap (e.g. the laptop slept overnight) the server
        # may refuse a very old resume point, and even if it accepted, hours of old posts would
        # be stamped "scored now" and show up as a fake spike. Rejoin the live stream instead.
        resume = None
        if state.cursor_us is not None:
            age_s = time.time() - state.cursor_us / 1_000_000
            if age_s > MAX_RESUME_AGE_S:
                say(f"last event was {age_s / 60:.0f} min ago (asleep or offline?); "
                    "too old to replay, rejoining the live stream")
                state.cursor_us = None
            elif resume_failures >= MAX_RESUME_FAILURES:
                say(f"resuming failed {resume_failures} times in a row; rejoining the live stream instead")
                state.cursor_us, resume_failures = None, 0
            else:
                resume = state.cursor_us - RESUME_REWIND_US
        url = build_url(endpoint, resume)
        got_frame = False
        try:
            with connect(url, subprotocols=[SUBPROTOCOL], open_timeout=10, max_size=2 ** 22) as ws:
                state.endpoint, state.connected = endpoint, True
                say(f"connected to {endpoint}" + (" (resuming from just before the last event seen)" if resume else ""))
                last_frame = time.monotonic()
                while not stop.is_set():
                    try:
                        frame = ws.recv(timeout=5)
                    except TimeoutError:
                        if time.monotonic() - last_frame > recv_timeout:
                            say(f"no data for {recv_timeout:.0f}s; reconnecting")
                            break
                        continue
                    last_frame = time.monotonic()
                    if not got_frame:
                        got_frame, attempt, resume_failures = True, 0, 0
                    yield frame
        except Exception as e:  # network errors of every kind -> retry
            say(f"connection problem ({type(e).__name__}: {e}); will retry")
            if resume is not None and not got_frame:
                resume_failures += 1
        finally:
            state.connected = False
        if stop.is_set():
            break
        stats.add(reconnects=1)
        attempt += 1
        delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** min(attempt, 5)) + random.uniform(0, 1)
        sleep_interruptible(stop, delay)


# ── Queue helpers ──────────────────────────────────────────
def put_drop_oldest(q, item, stats):
    """Keep the pipeline live: if scoring falls behind, discard the OLDEST
    waiting post rather than blocking the stream or growing without bound."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
                stats.add(dropped=1)
            except queue.Empty:
                pass


def collect_batch(q, max_items, window_s, done):
    batch = []
    deadline = time.monotonic() + window_s
    while len(batch) < max_items:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            batch.append(q.get(timeout=min(remaining, 0.25)))
        except queue.Empty:
            if done.is_set():      # source finished and nothing left to wait for
                break
    return batch


# ── Database ───────────────────────────────────────────────
COMMENT_COLS = (["run_id", "post_key", "created_at", "scored_at", "lag_s", "lang", "is_reply",
                 "char_len", "word_count", "upper_ratio"] + LABEL_COLS +
                ["top_label", "top_score", "flags_f1", "flags_hp", "preview"])
INSERT_SQL = (f"INSERT OR IGNORE INTO comments ({', '.join(COMMENT_COLS)}) "
              f"VALUES ({', '.join('?' * len(COMMENT_COLS))})")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    endpoint      TEXT,
    model_file    TEXT,
    settings_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL,
    post_key   TEXT NOT NULL UNIQUE,
    created_at TEXT,
    scored_at  TEXT NOT NULL,
    lag_s      REAL,
    lang       TEXT,
    is_reply   INTEGER,
    char_len   INTEGER,
    word_count INTEGER,
    upper_ratio REAL,
    {', '.join(f'{c} REAL' for c in LABEL_COLS)},
    top_label  TEXT,
    top_score  REAL,
    flags_f1   INTEGER,
    flags_hp   INTEGER,
    preview    TEXT
);
CREATE INDEX IF NOT EXISTS idx_comments_scored_at ON comments(scored_at);
CREATE TABLE IF NOT EXISTS stream_stats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    ts          TEXT NOT NULL,
    seen        INTEGER, passed INTEGER, scored INTEGER, dropped INTEGER,
    deletes     INTEGER, reconnects INTEGER, queue_depth INTEGER
);
CREATE INDEX IF NOT EXISTS idx_stats_ts ON stream_stats(ts);
"""


def init_db(path):
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")      # lets the dashboard read while we write
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def start_run(conn, endpoint, model_file, settings):
    cur = conn.execute(
        "INSERT INTO runs (started_at, endpoint, model_file, settings_json) VALUES (?, ?, ?, ?)",
        (utc_now_iso(), endpoint, model_file, json.dumps(settings)))
    conn.commit()
    return cur.lastrowid


def end_run(conn, run_id):
    conn.execute("UPDATE runs SET ended_at = ? WHERE run_id = ?", (utc_now_iso(), run_id))
    conn.commit()


def write_stats_row(conn, run_id, snap, queue_depth):
    conn.execute(
        "INSERT INTO stream_stats (run_id, ts, seen, passed, scored, dropped, deletes, reconnects, queue_depth) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, utc_now_iso(), snap["seen"], snap["passed"], snap["scored"], snap["dropped"],
         snap["deletes"], snap["reconnects"], queue_depth))
    conn.commit()


def apply_deletes(conn, delete_q, tombstones, stats, audit=None):
    keys = []
    while len(keys) < 5000:
        try:
            keys.append(delete_q.get_nowait())
        except queue.Empty:
            break
    if not keys:
        return 0
    for k in keys:
        tombstones.add(k)
    if audit is not None:
        audit.remove(keys)
    before = conn.total_changes
    conn.executemany("DELETE FROM comments WHERE post_key = ?", [(k,) for k in keys])
    conn.commit()
    removed = conn.total_changes - before
    if removed:
        stats.add(deletes=removed)
    return removed


def prune_old(conn, keep_hours):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=keep_hours)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    before = conn.total_changes
    conn.execute("DELETE FROM comments WHERE scored_at < ?", (cutoff,))
    conn.execute("DELETE FROM stream_stats WHERE ts < ?", (cutoff,))
    conn.commit()
    return conn.total_changes - before


# ── Scoring ────────────────────────────────────────────────
def make_rows(batch, probs, thr_f1, thr_hp, run_id, now_ts, now):
    rows, f1_count, hp_count, lag_sum = [], 0, 0, 0.0
    for item, p in zip(batch, probs):
        m_f1, m_hp = pack_mask(p >= thr_f1), pack_mask(p >= thr_hp)
        f1_count += m_f1 != 0
        hp_count += m_hp != 0
        j = int(np.argmax(p))
        lag = max(0.0, now - item["t_recv"])
        lag_sum += lag
        text = item["text"]
        rows.append((run_id, item["key"], item["created_at"], now_ts, round(lag, 3), item["lang"],
                     item["is_reply"], len(text), len(text.split()), round(upper_ratio(text), 4),
                     *[round(float(x), 5) for x in p], LABEL_COLS[j], round(float(p[j]), 5),
                     m_f1, m_hp, mask_preview(text)))
    return rows, f1_count, hp_count, lag_sum


# ── Pipeline ───────────────────────────────────────────────
def reader_main(frames, work_q, delete_q, stats, stop, done, state, cfg, say):
    try:
        for raw in frames:
            if stop.is_set():
                break
            cursor_us, kind, key, record = decode_frame(raw)
            stats.add(seen=1)
            if cursor_us is not None and (state.cursor_us is None or cursor_us > state.cursor_us):
                state.cursor_us = cursor_us
            if kind == "delete":
                delete_q.put(key)
            elif kind == "create":
                prepared = prepare_post(record, cfg["replies_only"], cfg["lang"])
                if prepared is not None:
                    prepared["key"], prepared["t_recv"] = key, time.time()
                    stats.add(passed=1)
                    put_drop_oldest(work_q, prepared, stats)
    except SystemExit as e:
        state.error = str(e)
    except Exception as e:
        state.error = f"{type(e).__name__}: {e}"
        say(f"reader stopped: {state.error}")
    finally:
        done.set()


def status_line(snap, prev, dt, queue_depth, mode):
    flagged = snap["flagged_hp"] if mode == "high_precision" else snap["flagged_f1"]
    scored_delta = snap["scored"] - prev["scored"]
    rate = scored_delta / dt if dt > 0 else 0.0
    lag = (snap["lag_sum"] - prev["lag_sum"]) / scored_delta if scored_delta else 0.0
    pct = 100.0 * flagged / max(1, snap["scored"])
    return (f"scored {snap['scored']:,} | flagged ({mode}) {flagged:,} ({pct:.2f}%) | "
            f"{rate:.1f}/s | lag {lag:.1f}s | queue {queue_depth} | dropped {snap['dropped']:,} | "
            f"deletes honored {snap['deletes']:,} | reconnects {snap['reconnects']}")


def run_pipeline(frames, cfg, model, thresholds, conn, run_id, say=log, batch_size=BATCH_SIZE,
                 queue_max=QUEUE_MAX, batch_window=BATCH_WINDOW_S, stats_interval=STATS_INTERVAL_S,
                 duration_s=None, audit=None):
    thr_f1 = threshold_vector(thresholds, "f1")
    thr_hp = threshold_vector(thresholds, "high_precision")
    work_q = queue.Queue(maxsize=queue_max)
    delete_q = queue.SimpleQueue()
    stop, done = threading.Event(), threading.Event()
    stats, state, tombstones = Stats(), StreamState(), Tombstones()
    if hasattr(frames, "bind"):               # live sources need the shared state/stats
        frames = frames.bind(stop, state, stats)

    reader = threading.Thread(target=reader_main, daemon=True, name="reader",
                              args=(frames, work_q, delete_q, stats, stop, done, state, cfg, say))
    reader.start()
    started = time.monotonic()
    last_stats = last_prune = started
    prev_snap = stats.snapshot()
    try:
        while True:
            batch = collect_batch(work_q, batch_size, batch_window, done)
            apply_deletes(conn, delete_q, tombstones, stats, audit)
            kept = [b for b in batch if b["key"] not in tombstones]
            if len(kept) != len(batch):       # deleted before we scored it: never stored
                stats.add(deletes=len(batch) - len(kept))
            batch = kept
            if batch:
                probs = predict_probs(model, make_inputs([b["text"] for b in batch]), batch_size=len(batch))
                ts = utc_now_iso()
                rows, f1_n, hp_n, lag_sum = make_rows(batch, probs, thr_f1, thr_hp, run_id, ts, time.time())
                if audit is not None:             # sample from everything flagged in Sensitive mode
                    for item, p in zip(batch, probs):
                        m_f1 = pack_mask(p >= thr_f1)
                        if m_f1:
                            audit.offer(item["key"], ts, item["text"], p, m_f1, pack_mask(p >= thr_hp))
                conn.executemany(INSERT_SQL, rows)
                conn.commit()
                stats.add(scored=len(rows), flagged_f1=f1_n, flagged_hp=hp_n, lag_sum=lag_sum)
            now = time.monotonic()
            if now - last_stats >= stats_interval:
                snap = stats.snapshot()
                write_stats_row(conn, run_id, snap, work_q.qsize())
                extra = f" | audit {audit.count()}/{audit.size}" if audit is not None else ""
                say(status_line(snap, prev_snap, now - last_stats, work_q.qsize(), cfg["mode"]) + extra)
                prev_snap, last_stats = snap, now
            if now - last_prune >= PRUNE_INTERVAL_S:
                n = prune_old(conn, cfg["keep_hours"])
                if n:
                    say(f"pruned {n:,} rows older than {cfg['keep_hours']}h")
                last_prune = now
            if done.is_set() and work_q.empty() and delete_q.empty() and not batch:
                break
            if duration_s is not None and now - started >= duration_s:
                say("duration reached; stopping")
                break
    except KeyboardInterrupt:
        say("Ctrl+C received; stopping ...")
    finally:
        stop.set()
        snap = stats.snapshot()
        write_stats_row(conn, run_id, snap, work_q.qsize())
        end_run(conn, run_id)
    if state.error:
        say(f"stream ended with an error: {state.error}")
    return stats.snapshot(), state


class LiveSource:
    """Wraps the websocket generator so run_pipeline can hand it the shared
    stop flag / state / stats once they exist."""

    def __init__(self, endpoints, say=log):
        self.endpoints, self.say = endpoints, say

    def bind(self, stop, state, stats):
        return websocket_frames(self.endpoints, stop, state, stats, self.say)


# ── Modes ──────────────────────────────────────────────────
def describe_structure(obj, depth=3):
    """Key names and value TYPES only -- never values -- so nothing from a
    post can be printed while still showing the message format."""
    if depth == 0:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {k: describe_structure(v, depth - 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [describe_structure(obj[0], depth - 1)] if obj else []
    return type(obj).__name__


def run_probe(seconds, endpoints):
    log(f"PROBE: connecting for {seconds}s (no model, nothing stored, no post text printed)")
    stop, state, stats = threading.Event(), StreamState(), Stats()
    timer = threading.Timer(seconds, stop.set)
    timer.daemon = True
    timer.start()
    counts = Counter()
    shapes = []
    for raw in websocket_frames(endpoints, stop, state, stats):
        counts["frames"] += 1
        cursor_us, kind, key, record = decode_frame(raw)
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            counts["not_json"] += 1
            continue
        if len(shapes) < 3:
            shapes.append(describe_structure(msg))
        payload = msg.get("payload") if isinstance(msg, dict) else None
        counts["env:" + str(msg.get("$type")) if isinstance(msg, dict) else "env:?"] += 1
        if isinstance(payload, dict):
            counts["payload:" + str(payload.get("$type"))] += 1
            if payload.get("collection"):
                counts["collection:" + str(payload.get("collection"))] += 1
        if cursor_us is not None:
            counts["with_parsable_time"] += 1
        if kind == "create":
            counts["post_creates"] += 1
            if prepare_post(record, replies_only=False, lang=None):
                counts["post_creates_usable_text"] += 1
            if isinstance(record.get("reply"), dict):
                counts["replies"] += 1
            if prepare_post(record, replies_only=True, lang="en"):
                counts["english_replies_pass_filters"] += 1
        elif kind == "delete":
            counts["post_deletes"] += 1
    timer.cancel()
    print("\n--- probe results ---")
    print(f"endpoint used: {state.endpoint}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print("structure of the first messages (key names and value types only):")
    for s in shapes:
        print("  " + json.dumps(s))
    if counts["post_creates"] > 0 and counts["english_replies_pass_filters"] > 0:
        rate = counts["english_replies_pass_filters"] / max(1, seconds)
        print(f"\nPROBE OK: posts decode correctly; about {rate:.1f} English replies/second would reach the model.")
        return 0
    if counts["frames"] == 0:
        print("\nPROBE FAILED: no frames received. Check the connection messages above "
              "(firewall/VPN/proxy can block WebSockets).")
    else:
        print("\nPROBE WARNING: frames arrived but no usable posts decoded. The message format may differ "
              "from the docs -- paste this whole output so the decoder can be fixed.")
    return 1


# Synthetic events for --selftest (deliberately mild: no slurs, nothing graphic).
_NEUTRAL = [
    "Thanks for sharing this, it was genuinely helpful and well explained.",
    "I read the article yesterday and found the second section the most interesting.",
    "Does anyone know a good place to start learning about this topic?",
    "Congratulations on the new job, you absolutely deserve it!",
    "I think the schedule changes will make the commute a lot easier for everyone.",
    "That recipe turned out great, my whole family asked for seconds.",
    "Interesting point, though I would love to see the data behind it.",
    "The concert was amazing and the sound quality was better than expected.",
]
_HOSTILE = [
    "You are such an idiot and nobody wants to hear your stupid opinion.",
    "Shut up already, you clueless moron, nobody asked you.",
    "What a pathetic loser, go away and stop wasting everyone's time.",
    "I hate you and your useless garbage takes, you total fool.",
]


def make_synthetic(n_posts=120):
    """Returns (frames, truth). truth = {'expected_keys': set, 'hostile_keys': set, 'neutral_keys': set,
    'deleted_keys': set, 'sentinels': [...]}."""
    frames, seq = [], 1000
    truth = {"expected_keys": set(), "hostile_keys": set(), "neutral_keys": set(),
             "deleted_keys": set(), "sentinels": []}
    rng = random.Random(7)

    def env(payload):
        return json.dumps({"$type": "message", "payload": payload})

    def create(i, text, langs=("en",), reply=True, did=None, rkey=None):
        nonlocal seq
        seq += 1
        did = did or f"did:plc:selftest{i % 9}"
        rkey = rkey or f"3selftest{i:05d}"
        record = {"$type": COLLECTION, "createdAt": "2026-01-01T00:00:00.000Z", "text": text, "langs": list(langs)}
        if reply:
            record["reply"] = {"root": {"uri": "at://x/y/z", "cid": "c"}, "parent": {"uri": "at://x/y/z", "cid": "c"}}
        payload = {"$type": "network.bsky.jetstream.subscribeEvents#commit", "did": did, "seq": seq,
                   "time": "2026-01-01T00:00:00.000000Z", "operation": "create", "collection": COLLECTION,
                   "rkey": rkey, "rev": "r", "cid": "c", "record": record}
        return env(payload), post_key(did, rkey), (did, rkey)

    def delete(did, rkey):
        nonlocal seq
        seq += 1
        return env({"$type": "network.bsky.jetstream.subscribeEvents#commit", "did": did, "seq": seq,
                    "time": "2026-01-01T00:00:00.000000Z", "operation": "delete",
                    "collection": COLLECTION, "rkey": rkey, "rev": "r"})

    for i in range(n_posts):
        hostile = rng.random() < 0.3
        base = rng.choice(_HOSTILE if hostile else _NEUTRAL)
        text = f"{base} (take {i})"
        frame, key, ids = create(i, text)
        frames.append(frame)
        truth["expected_keys"].add(key)
        (truth["hostile_keys"] if hostile else truth["neutral_keys"]).add(key)
        if i % 15 == 3:                       # delete shortly AFTER the post (race path)
            frames.append(delete(*ids))
            truth["deleted_keys"].add(key)
        elif i % 15 == 9 and i > 20:          # delete much later (stored-row path)
            truth.setdefault("late_delete", []).append((i, ids, key))
        if i % 20 == 5:                       # at-least-once duplicate delivery
            frames.append(frame)
    for i, ids, key in truth.get("late_delete", []):
        frames.append(delete(*ids))
        truth["deleted_keys"].add(key)
    # ineligible traffic that must be ignored
    for j in range(8):
        frames.append(create(900 + j, "Esto es una respuesta en otro idioma completamente.", langs=("es",))[0])
        frames.append(create(920 + j, "A top level post that is not a reply at all.", reply=False)[0])
        frames.append(create(940 + j, "ok")[0])
    frames += ["this is not json", '{"foo": 1}', "[1, 2, 3]", "null",
               env({"$type": "network.bsky.jetstream.subscribeEvents#identity", "did": "did:plc:x", "seq": 5}),
               env({"$type": "x#commit", "did": "did:plc:x", "seq": 6, "operation": "create",
                    "collection": "app.bsky.feed.like", "rkey": "k", "record": {"subject": {}}})]
    truth["expected_keys"] -= truth["deleted_keys"]
    truth["hostile_keys"] -= truth["deleted_keys"]
    truth["neutral_keys"] -= truth["deleted_keys"]
    truth["sentinels"] = ["helpful", "article", "recipe", "concert", "idiot", "moron", "loser",
                          "garbage", "schedule", "congratulations"]
    return frames, truth


def run_selftest(model_dir):
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(SELFTEST_DB + suffix)
        except OSError:
            pass
    model, thresholds = load_artifacts_checked(model_dir)
    log("SELFTEST: warming up the model ...")
    score_texts(model, ["warm up"])
    frames, truth = make_synthetic()
    cfg = {"replies_only": True, "lang": "en", "mode": "high_precision", "keep_hours": 48}
    conn = init_db(SELFTEST_DB)
    run_id = start_run(conn, "selftest", thresholds["model_file"], {"selftest": True})
    log(f"SELFTEST: feeding {len(frames)} synthetic frames through the pipeline ...")
    snap, _ = run_pipeline(iter(frames), cfg, model, thresholds, conn, run_id,
                           batch_size=16, batch_window=0.3, stats_interval=1.0)
    conn.close()

    conn = sqlite3.connect(SELFTEST_DB)
    keys = {r[0] for r in conn.execute("SELECT post_key FROM comments")}
    n_rows = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    text_cols = " || ' ' || ".join(f"COALESCE({c}, '')" for c in
                                   ["preview", "lang", "top_label", "post_key", "created_at"])
    blob = " ".join(r[0] for r in conn.execute(f"SELECT {text_cols} FROM comments")).lower()
    runs_ended = conn.execute("SELECT COUNT(*) FROM runs WHERE ended_at IS NOT NULL").fetchone()[0]
    stat_rows = conn.execute("SELECT COUNT(*) FROM stream_stats").fetchone()[0]

    def mean_top(keyset, label_col="top_score"):
        marks = ",".join("?" * len(keyset)) or "''"
        row = conn.execute(f"SELECT AVG({label_col}) FROM comments WHERE post_key IN ({marks})", list(keyset)).fetchone()
        return row[0] or 0.0

    hostile_mean, neutral_mean = mean_top(truth["hostile_keys"]), mean_top(truth["neutral_keys"])
    preview_has_letters = any(ch.isalnum() for (p,) in conn.execute("SELECT preview FROM comments") for ch in p)
    conn.close()

    checks = [
        ("every eligible, undeleted post stored exactly once (dedupe works)", keys == truth["expected_keys"] and n_rows == len(truth["expected_keys"])),
        ("deleted posts absent, including delete-before-score race", not (keys & truth["deleted_keys"])),
        ("ineligible traffic (non-English, non-reply, too short, junk) ignored", keys <= truth["expected_keys"]),
        ("no post text anywhere in the database", not any(w in blob for w in truth["sentinels"])),
        ("previews fully masked (no letters/digits survive)", not preview_has_letters),
        ("run closed and stats rows written", runs_ended == 1 and stat_rows >= 1),
        ("nothing dropped at this volume", snap["dropped"] == 0),
    ]
    print("\n--- selftest results ---")
    ok = True
    for name, passed in checks:
        ok &= bool(passed)
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n  stored {n_rows} comments | flagged (high-precision) {snap['flagged_hp']} | "
          f"flagged (F1-optimal) {snap['flagged_f1']} | deletes honored {snap['deletes']}")
    direction = "as expected" if hostile_mean > neutral_mean else "NOT as expected -- worth a look"
    print(f"  model sanity: mean top score hostile-style {hostile_mean:.3f} vs neutral-style {neutral_mean:.3f} ({direction})")
    print("  (that last line judges the model, not the pipeline; the checks above judge the pipeline)")
    print("\nSELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def load_artifacts_checked(model_dir):
    if not os.path.exists(os.path.join(model_dir, "thresholds.json")):
        raise SystemExit(f"No trained model found in {model_dir}/. Run tox_phase2_train.py first.")
    model, thresholds = load_artifacts(model_dir)
    if thresholds.get("label_cols") != LABEL_COLS:
        raise SystemExit("The saved model's label order doesn't match tox_common.LABEL_COLS; retrain.")
    return model, thresholds


def run_live(args):
    model, thresholds = load_artifacts_checked(args.model_dir)
    log("warming up the model ...")
    score_texts(model, ["warm up"])
    endpoints = [args.endpoint] if args.endpoint else ENDPOINTS
    cfg = {"replies_only": not args.include_posts, "lang": None if args.lang == "any" else args.lang,
           "mode": args.mode, "keep_hours": args.keep_hours}
    can_flag = [l for l in LABEL_COLS if thresholds["high_precision_threshold"].get(l) is not None]
    never = [l for l in LABEL_COLS if l not in can_flag]
    log(f"model: {thresholds['model_file']} | database: {args.db} | keeping {args.keep_hours}h of rows")
    log(f"filters: {'replies only' if cfg['replies_only'] else 'all posts'}, "
        f"language {'any' if cfg['lang'] is None else cfg['lang']}")
    log(f"high-precision mode can flag: {', '.join(can_flag) or 'nothing'}"
        + (f" (never flags: {', '.join(never)})" if never else ""))
    audit = None
    if args.audit:
        audit = AuditReservoir(args.audit_db, args.audit, reset=args.audit_reset)
        log(f"AUDIT MODE ON: the text of up to {args.audit} randomly chosen flagged replies will be stored LOCALLY in "
            f"{args.audit_db} (no account names). The monitoring database still stores no text.")
        log(f"audit sample so far: {audit.count()}/{args.audit}. Review it later with: python .\\tox_audit.py")
    conn = init_db(args.db)
    settings = {**cfg, "label_cols": LABEL_COLS, "f1_threshold": thresholds["f1_threshold"],
                "high_precision_threshold": thresholds["high_precision_threshold"],
                "bitmask_note": "bit j of flags_f1/flags_hp = label j in label_cols crossed its threshold"}
    run_id = start_run(conn, ",".join(endpoints), thresholds["model_file"], settings)
    log(f"run {run_id} started. Press Ctrl+C to stop.")
    duration = args.duration_min * 60 if args.duration_min else None
    try:
        snap, state = run_pipeline(LiveSource(endpoints), cfg, model, thresholds, conn, run_id,
                                   duration_s=duration, audit=audit)
    finally:
        conn.close()
        if audit is not None:
            kept = audit.count()
            audit.close()
            log(f"audit sample: {kept}/{args.audit} comments saved in {args.audit_db}. "
                "Stop collecting, then review with:  python .\\tox_audit.py")
    log(f"done. scored {snap['scored']:,} comments this run; database: {args.db}")
    return 1 if state.error else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Score the live Bluesky stream for toxicity.")
    p.add_argument("--probe", nargs="?", const=15, type=int, metavar="SECONDS",
                   help="network check only (default 15 s); no model, nothing stored")
    p.add_argument("--selftest", action="store_true", help="offline end-to-end test with synthetic events")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--mode", choices=["high_precision", "f1"], default="high_precision",
                   help="which thresholds the console status line counts as 'flagged' (the database stores both)")
    p.add_argument("--lang", default="en", help="language code to keep, or 'any'")
    p.add_argument("--include-posts", action="store_true", help="also score top-level posts, not just replies")
    p.add_argument("--keep-hours", type=float, default=48.0)
    p.add_argument("--duration-min", type=float, default=None)
    p.add_argument("--endpoint", default=None, help="override the Jetstream endpoint (wss://...)")
    p.add_argument("--audit", type=int, default=0, metavar="N",
                   help="also keep the text of up to N random flagged replies locally for hand review (tox_audit.py)")
    p.add_argument("--audit-db", default=DEFAULT_AUDIT_DB)
    p.add_argument("--audit-reset", action="store_true", help="discard any existing audit sample and start over")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.probe is not None:
        return run_probe(args.probe, [args.endpoint] if args.endpoint else ENDPOINTS)
    if args.selftest:
        return run_selftest(args.model_dir)
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
