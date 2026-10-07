"""
=============================================================
Real-Time Toxic Content Moderation
Phase 4: Live Dashboard (Dash)
=============================================================
Author : Setu Patel
Requires: tox_stream.db (written by tox_phase3_stream.py)  and:  pip install dash

    python .\\tox_phase4_dashboard.py            then open http://127.0.0.1:8050
    python .\\tox_phase4_dashboard.py --check    build every panel once, no server

Run the collector (tox_phase3_stream.py) in one terminal and this dashboard
in another. They only share the SQLite file, so either can be stopped and
restarted without affecting the other.

What the panels show
--------------------
* Status + KPIs        comments scored, % flagged, throughput, lag, drops,
                       deletions honored, reconnects.
* Flag rate over time  comments per time bucket (bars) and % flagged (line).
* Flags by category    which labels the model is flagging -- if most flags
                       are "obscene" alone, the flag rate mostly measures
                       swearing, not abuse.
* Score distribution   how confident the model is, flagged vs not (log scale).
* Live feed            newest comments: category + confidence + a fully
                       masked preview. No comment text exists anywhere in
                       the database, so none can be shown.
* Limitations          the model's measured test performance, and why the
                       live flag rate is NOT an estimate of how toxic
                       Bluesky is.

Two modes: "Strict" counts a comment as flagged only at the high-precision
thresholds (fewer false alarms, misses more); "Sensitive" uses the F1-optimal
thresholds (catches more, more false alarms).
=============================================================
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from tox_common import LABEL_COLS

# ── Config ─────────────────────────────────────────────────
DEFAULT_DB = "tox_stream.db"
TS_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"          # must match tox_phase3_stream.py
LIVE_IF_NEWER_THAN_S = 30
FEED_ROWS = 25
HIST_BINS = 20

# `bucket` = how many leading characters of the ISO timestamp define a time bucket:
# 16 -> minute, 15 -> ten minutes, 13 -> hour. Grouping on the string prefix avoids
# parsing a million timestamps in SQL.
WINDOWS = {
    "15m": {"label": "Last 15 min", "minutes": 15, "bucket": 16, "refresh_ms": 5_000},
    "1h": {"label": "Last hour", "minutes": 60, "bucket": 16, "refresh_ms": 10_000},
    "6h": {"label": "Last 6 hours", "minutes": 360, "bucket": 15, "refresh_ms": 30_000},
    "24h": {"label": "Last 24 hours", "minutes": 1440, "bucket": 15, "refresh_ms": 60_000},
    "all": {"label": "All stored", "minutes": None, "bucket": 13, "refresh_ms": 60_000},
}
MODES = {
    "strict": {"col": "flags_hp", "label": "Strict (high precision)", "thr_key": "high_precision_threshold"},
    "sensitive": {"col": "flags_f1", "label": "Sensitive (F1-optimal)", "thr_key": "f1_threshold"},
}

PAPER, CARD, INK, SOFT = "#EEF0EA", "#F7F8F4", "#16211B", "#47554B"
LINE, AMBER, TEAL, GREEN, RED, MUTED = "#C7CCBE", "#C8862E", "#234A56", "#3C6E4B", "#A2453A", "#B8BDB0"
LABEL_COLORS = {"toxic": RED, "severe_toxic": "#6E2A22", "obscene": AMBER,
                "threat": "#7A3B8F", "insult": TEAL, "identity_hate": "#4B6A2E"}


# ── Time helpers ───────────────────────────────────────────
def utc_now():
    return datetime.now(timezone.utc)


def window_cutoff(minutes):
    """ISO string for 'now - minutes' ('' means no lower bound: everything)."""
    if minutes is None:
        return ""
    return (utc_now() - timedelta(minutes=minutes)).strftime(TS_FMT)


def parse_ts(s):
    return datetime.strptime(s, TS_FMT).replace(tzinfo=timezone.utc)


def bucket_to_local(key):
    """Bucket key ('2026-10-05T22:3' etc.) -> naive LOCAL wall-clock datetime.
    Plotly shows the wall-clock part of a timestamp and ignores any offset, so
    we convert to local time ourselves and drop the timezone."""
    pad = {13: ":00:00", 15: "0:00", 16: ":00"}[len(key)]
    dt = datetime.strptime(key + pad, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return dt.astimezone().replace(tzinfo=None)


def fmt_int(n):
    return f"{int(n):,}"


# ── Data layer (plain sqlite + pandas; no Dash, no plotly) ─
def open_db(path):
    """Read-only-in-practice connection. Returns None if the file doesn't exist
    (we never want to create an empty database by accident)."""
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path, timeout=5, isolation_level=None)   # we manage transactions explicitly
    conn.execute("PRAGMA query_only = ON")
    return conn


def query_kpis(conn, cutoff, flag_col):
    n, flagged, avg_lag, first, last = conn.execute(
        f"SELECT COUNT(*), COALESCE(SUM({flag_col} != 0), 0), AVG(lag_s), MIN(scored_at), MAX(scored_at) "
        "FROM comments WHERE scored_at >= ?", (cutoff,)).fetchone()
    rate = None
    if n and first and last:
        span = (parse_ts(last) - parse_ts(first)).total_seconds()
        rate = n / span if span >= 1 else None
    return {"n": n or 0, "flagged": flagged or 0,
            "flagged_pct": (100.0 * flagged / n) if n else 0.0,
            "avg_lag": avg_lag, "rate_per_s": rate, "first": first, "last": last}


_BUCKET_MINUTES = {16: 1, 15: 10, 13: 60}


def _bucket_key_to_utc(key):
    pad = {13: ":00:00", 15: "0:00", 16: ":00"}[len(key)]
    return datetime.strptime(key + pad, "%Y-%m-%dT%H:%M:%S")


def fill_gaps(df, bucket_len):
    """Adds an explicit zero-count row for every bucket between the first and last
    that has no data, so an outage shows up as a visible break instead of the
    line quietly bridging it."""
    if len(df) < 2:
        return df
    step = timedelta(minutes=_BUCKET_MINUTES[bucket_len])
    cur, end = _bucket_key_to_utc(df["key"].iloc[0]), _bucket_key_to_utc(df["key"].iloc[-1])
    keys = []
    while cur <= end:
        keys.append(cur.strftime("%Y-%m-%dT%H:%M")[:bucket_len])
        cur += step
    full = pd.DataFrame({"key": keys}).merge(df, on="key", how="left")
    full["n"] = full["n"].fillna(0).astype(int)
    full["flagged"] = full["flagged"].fillna(0).astype(int)
    return full


def time_axis_range(minutes):
    """[start, end] of the chosen window in local wall-clock time, so the time
    chart always spans the whole window and missing recent data is visible."""
    if minutes is None:
        return None
    end = utc_now().astimezone().replace(tzinfo=None)
    fmt = "%Y-%m-%d %H:%M:%S"
    return [(end - timedelta(minutes=minutes)).strftime(fmt), end.strftime(fmt)]


_TICK_STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]


def tick_decimals(step):
    """Decimal places needed so every tick label is distinct (2.5 -> 1, 0.25 -> 2, 5 -> 0)."""
    for dec in (0, 1, 2):
        if abs(step * 10 ** dec - round(step * 10 ** dec)) < 1e-9:
            return dec
    return 3


def nice_axis(max_value, ticks=5):
    """(step, top) giving round tick values (0, 2.5, 5, ...) instead of whatever
    values happen to line up with another axis's gridlines."""
    if max_value is None or (isinstance(max_value, float) and math.isnan(max_value)) or max_value <= 0:
        return 1.0, 5.0
    raw = max_value * 1.05 / ticks
    step = next((s for s in _TICK_STEPS if s >= raw), 100)
    return step, step * math.ceil(max_value * 1.05 / step)


def query_timeseries(conn, cutoff, flag_col, bucket_len):
    rows = conn.execute(
        f"SELECT substr(scored_at, 1, {int(bucket_len)}) AS b, COUNT(*), COALESCE(SUM({flag_col} != 0), 0) "
        "FROM comments WHERE scored_at >= ? GROUP BY b ORDER BY b", (cutoff,)).fetchall()
    current = utc_now().strftime(TS_FMT)[:bucket_len]
    # The newest bucket is still filling up; plotting it would show a fake dip.
    rows = [r for r in rows if r[0] != current]
    df = pd.DataFrame(rows, columns=["key", "n", "flagged"])
    if df.empty:
        return df.assign(t=pd.Series(dtype="datetime64[ns]"), pct=pd.Series(dtype=float))
    df = fill_gaps(df, bucket_len)
    df["t"] = [bucket_to_local(k) for k in df["key"]]
    df["pct"] = np.where(df["n"] > 0, 100.0 * df["flagged"] / df["n"].where(df["n"] > 0, 1), np.nan)
    return df


def query_categories(conn, cutoff, flag_col):
    exprs = ", ".join(f"COALESCE(SUM(({flag_col} >> {j}) & 1), 0)" for j in range(len(LABEL_COLS)))
    row = conn.execute(f"SELECT {exprs} FROM comments WHERE scored_at >= ?", (cutoff,)).fetchone()
    return dict(zip(LABEL_COLS, [int(v) for v in row]))


def query_hist(conn, cutoff, flag_col):
    rows = conn.execute(
        f"SELECT MIN(CAST(top_score * {HIST_BINS} AS INTEGER), {HIST_BINS - 1}) AS b, ({flag_col} != 0) AS f, COUNT(*) "
        "FROM comments WHERE scored_at >= ? GROUP BY b, f", (cutoff,)).fetchall()
    flagged = np.zeros(HIST_BINS, dtype=int)
    unflagged = np.zeros(HIST_BINS, dtype=int)
    for b, f, c in rows:
        (flagged if f else unflagged)[int(b)] += int(c)
    return flagged, unflagged


def mask_to_labels(mask):
    return [LABEL_COLS[j] for j in range(len(LABEL_COLS)) if (int(mask) >> j) & 1]


def query_feed(conn, flag_col, n_rows):
    cols = ", ".join(LABEL_COLS)
    rows = conn.execute(
        f"SELECT scored_at, {cols}, top_label, top_score, {flag_col}, word_count, upper_ratio, preview "
        "FROM comments ORDER BY id DESC LIMIT ?", (n_rows,)).fetchall()
    k = len(LABEL_COLS)
    feed = []
    for r in rows:
        scores = dict(zip(LABEL_COLS, r[1:1 + k]))
        top_label, top_score, mask, words, caps, preview = r[1 + k:]
        labels = mask_to_labels(mask)
        feed.append({
            "time": parse_ts(r[0]).astimezone().strftime("%H:%M:%S"),
            "flagged": bool(labels),
            "category": ", ".join(labels) if labels else "-",
            "confidence": max(scores[l] for l in labels) if labels else top_score,
            "top_label": top_label, "words": words, "caps": caps, "preview": preview or "",
        })
    return feed


def query_health(conn):
    run = conn.execute("SELECT run_id, started_at, ended_at, settings_json FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    stats = conn.execute("SELECT dropped, deletes, reconnects FROM stream_stats ORDER BY id DESC LIMIT 1").fetchone()
    settings = {}
    if run and run[3]:
        try:
            settings = json.loads(run[3])
        except ValueError:
            settings = {}
    return {"run_id": run[0] if run else None, "started_at": run[1] if run else None,
            "ended_at": run[2] if run else None, "settings": settings,
            "dropped": stats[0] if stats else 0, "deletes": stats[1] if stats else 0,
            "reconnects": stats[2] if stats else 0}


def load_dashboard_data(db_path, window_key, mode_key):
    """Everything a refresh needs, as plain Python/pandas objects.
    state is 'no_db', 'empty' (tables exist, no rows yet) or 'ok'."""
    conn = open_db(db_path)
    if conn is None:
        return {"state": "no_db"}
    try:
        w, col = WINDOWS[window_key], MODES[mode_key]["col"]
        cutoff = window_cutoff(w["minutes"])
        # ONE read transaction for the whole refresh. In WAL mode every query inside it sees
        # the same frozen snapshot, so the panels always describe the same set of comments.
        # (Without this, a batch landing between two queries made one panel count comments
        # another had not seen -- e.g. more 'toxic' flags than flagged comments.)
        conn.execute("BEGIN")
        try:
            total = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        except sqlite3.OperationalError:
            return {"state": "no_db"}
        health = query_health(conn)
        if total == 0:
            return {"state": "empty", "health": health}
        kpis = query_kpis(conn, cutoff, col)
        last_any = conn.execute("SELECT MAX(scored_at) FROM comments").fetchone()[0]
        age = (utc_now() - parse_ts(last_any)).total_seconds()
        return {
            "state": "ok", "kpis": kpis, "health": health, "age_s": age,
            "ts": query_timeseries(conn, cutoff, col, w["bucket"]),
            "cats": query_categories(conn, cutoff, col),
            "hist": query_hist(conn, cutoff, col),
            "feed": query_feed(conn, col, FEED_ROWS),
            "window_key": window_key, "mode_key": mode_key,
            "x_range": time_axis_range(w["minutes"]),
        }
    finally:
        conn.close()


# ── Figures (plain dicts; validated through plotly by --check) ─
def _base_layout(**extra):
    layout = {"paper_bgcolor": CARD, "plot_bgcolor": CARD, "font": {"color": INK, "size": 12},
              "margin": {"l": 56, "r": 56, "t": 36, "b": 44}, "legend": {"orientation": "h", "y": 1.12, "x": 0},
              "hovermode": "x unified"}
    layout.update(extra)
    return layout


def fig_timeseries(data):
    df = data["ts"]
    x = [t.strftime("%Y-%m-%d %H:%M:%S") for t in df["t"]]
    pct = [None if pd.isna(v) else round(float(v), 2) for v in df["pct"]]     # None = a gap in the line
    step, top = nice_axis(df["pct"].max() if len(df) else None)
    fig = {
        "data": [
            {"type": "bar", "x": x, "y": [int(v) for v in df["n"]], "name": "comments scored",
             "marker": {"color": MUTED}, "yaxis": "y"},
            {"type": "scatter", "mode": "lines+markers", "x": x, "y": pct, "connectgaps": False,
             "name": "% flagged", "yaxis": "y2", "line": {"color": RED, "width": 2}, "marker": {"size": 5}},
        ],
        "layout": _base_layout(
            yaxis={"title": {"text": "comments per bucket", "standoff": 10}, "gridcolor": LINE, "automargin": True},
            yaxis2={"title": {"text": "% flagged", "standoff": 10}, "overlaying": "y", "side": "right",
                    "tickmode": "linear", "tick0": 0, "dtick": step, "range": [0, top],
                    "ticksuffix": "%", "tickformat": f".{tick_decimals(step)}f",
                    "showgrid": False, "automargin": True},
            xaxis={"gridcolor": LINE}),
    }
    if data.get("x_range"):
        fig["layout"]["xaxis"]["range"] = data["x_range"]
    return fig


def fig_categories(data):
    settings = data["health"].get("settings", {})
    thr = settings.get(MODES[data["mode_key"]]["thr_key"], {})
    cats = data["cats"]
    never = [l for l in LABEL_COLS if thr and thr.get(l) is None]
    order = list(reversed(LABEL_COLS))
    ylabels = [l + (" *" if l in never else "") for l in order]
    fig = {
        "data": [{"type": "bar", "orientation": "h", "x": [cats[l] for l in order], "y": ylabels,
                  "marker": {"color": [LABEL_COLORS[l] for l in order]},
                  "text": [fmt_int(cats[l]) for l in order], "textposition": "outside", "cliponaxis": False,
                  "hovertemplate": "%{y}: %{x:,}<extra></extra>"}],
        "layout": _base_layout(xaxis={"gridcolor": LINE, "title": {"text": "comments flagged with this label"}},
                               yaxis={"automargin": True, "ticks": "outside", "ticklen": 10, "tickcolor": CARD},
                               hovermode="closest", showlegend=False,
                               margin={"l": 110, "r": 40, "t": 24, "b": 50}),
    }
    if never:
        fig["layout"]["annotations"] = [{"text": "* never flags in this mode (couldn't reach the precision target)",
                                         "xref": "paper", "yref": "paper", "x": 0, "y": -0.28,
                                         "showarrow": False, "font": {"size": 11, "color": SOFT}}]
    return fig


def fig_hist(data):
    flagged, unflagged = data["hist"]
    centers = [round((i + 0.5) / HIST_BINS, 3) for i in range(HIST_BINS)]
    return {
        "data": [
            {"type": "bar", "x": centers, "y": [int(v) for v in unflagged], "name": "not flagged", "marker": {"color": MUTED}},
            {"type": "bar", "x": centers, "y": [int(v) for v in flagged], "name": "flagged", "marker": {"color": RED}},
        ],
        "layout": _base_layout(barmode="overlay",
                               yaxis={"type": "log", "dtick": 1, "automargin": True, "gridcolor": LINE,
                                      "title": {"text": "comments (log scale)", "standoff": 10}},
                               xaxis={"title": {"text": "model's highest label score"}, "range": [0, 1]}),
    }


# ── Static text: limitations panel ─────────────────────────
def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def limitation_lines(models_dir, audit_path):
    """(plain strings) -- numbers come from the saved Phase 2 report when present."""
    m = load_json(os.path.join(models_dir, "phase2_metrics.json")) or {}
    strict = m.get("test_any_label_high_precision_thresholds") or {}
    sens = m.get("test_any_label_f1_thresholds") or {}
    lines = []
    if strict.get("precision") is not None and sens.get("precision") is not None:
        lines.append(
            f"On held-out Wikipedia comments, Strict mode flagged with {strict['precision']:.0%} precision and "
            f"{strict['recall']:.0%} recall; Sensitive mode with {sens['precision']:.0%} precision and "
            f"{sens['recall']:.0%} recall. Those figures describe Wikipedia talk pages, not Bluesky.")
    else:
        lines.append("Test-set precision/recall figures are read from models/phase2_metrics.json (not found).")
    hand_check = audit_lines(load_json(audit_path))
    measured = any(l.startswith("Hand-checked") for l in hand_check)
    lines += [
        "The model was trained on Wikipedia talk-page comments. Bluesky replies are shorter and use different "
        "slang, humor and casual profanity, so accuracy here is "
        + ("lower; the hand-check below measures by how much." if measured
           else "probably lower and false alarms more common."),
        "'Flagged' means the model's score crossed a threshold. It is not a verified violation, and the flag "
        "rate is not an estimate of how toxic Bluesky is. The category panel shows which labels fired; "
        + ("the hand-check below shows how many flags were genuinely abusive."
           if measured else "only a hand-check can show how many flags are genuinely abusive."),
        "Rare labels (threat, identity_hate, severe_toxic) are weak: they had too few training examples to "
        "reach the precision target, so Strict mode never flags them.",
        "Privacy by design: post text is never stored in this database or shown here (previews are fully masked), "
        "no account names are kept, and posts that get deleted on Bluesky are deleted from this database too.",
    ]
    lines += hand_check
    return lines


def audit_lines(audit):
    """Lines describing the hand-check written by tox_audit.py (empty if none yet)."""
    if not audit:
        return []
    out = []
    modes = audit.get("modes")
    if modes:
        for key, name in (("strict", "Strict"), ("sensitive", "Sensitive")):
            b = modes.get(key) or {}
            if b.get("n", 0) >= 10:
                lo, hi = b["abusive_ci"]
                out.append(
                    f"Hand-checked on Bluesky ({name} mode, {b['n']} flagged replies reviewed): "
                    f"{b['abusive_rate']:.0%} were abusive (95% CI {lo:.0%}-{hi:.0%}), "
                    f"{b['profane_only'] / b['n']:.0%} were profanity aimed at no one, and "
                    f"{b['not_toxic'] / b['n']:.0%} were not toxic at all. "
                    f"In total, {b['offensive_rate']:.0%} involved abusive or profane language.")
        if out:
            size = audit.get("sample_size")
            sample_note = (f" The sample was {size} replies flagged in Sensitive mode; the Strict figures use the "
                           "ones Strict mode also flags." if size else "")
            out.append("That check was done by one reviewer, blind to the model's scores and without the post being "
                       "replied to. It measures precision only: it says nothing about what the model misses." + sample_note)
    elif audit.get("n_reviewed"):                  # older summary format
        out.append(f"Hand-checked sample: of {audit['n_reviewed']} flagged Bluesky replies reviewed, "
                   f"{audit['precision']:.0%} were genuinely abusive "
                   f"(95% CI {audit['ci_low']:.0%}-{audit['ci_high']:.0%}).")
    return out


# ── Dash layer ─────────────────────────────────────────────
INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  body { margin: 0; background: __PAPER__; color: __INK__; font-family: Inter, system-ui, -apple-system, 'Segoe UI', sans-serif; }
  .wrap { max-width: 1280px; margin: 0 auto; padding: 24px 28px 48px; }
  h1 { font-family: Fraunces, Georgia, serif; font-weight: 500; font-size: 28px; margin: 0 0 4px; }
  .sub { color: __SOFT__; font-size: 14px; margin-bottom: 18px; }
  .row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 14px; }
  .card { background: __CARD__; border: 1px solid __LINE__; border-radius: 8px; padding: 14px 16px; }
  .kpi .v { font-family: Fraunces, Georgia, serif; font-size: 26px; }
  .kpi .l { color: __SOFT__; font-size: 12.5px; margin-top: 2px; }
  .kpi .s { color: __SOFT__; font-size: 11.5px; margin-top: 4px; }
  .grid2 { display: grid; grid-template-columns: 3fr 2fr; gap: 14px; margin-bottom: 14px; }
  .grid2b { display: grid; grid-template-columns: 2fr 3fr; gap: 14px; margin-bottom: 14px; }
  @media (max-width: 900px) { .grid2, .grid2b { grid-template-columns: 1fr; } }
  .card h3 { font-family: Fraunces, Georgia, serif; font-weight: 500; font-size: 17px; margin: 0 0 6px; }
  .controls label { margin-right: 14px; font-size: 13.5px; }
  .controls .group { margin-right: 28px; }
  .controls .t { font-size: 12px; color: __SOFT__; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 4px; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12.5px; font-weight: 600; }
  .live { background: #DDEBDD; color: __GREEN__; } .stale { background: #F1E2C8; color: #7A5316; }
  table.feed { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  table.feed th { text-align: left; color: __SOFT__; font-weight: 500; border-bottom: 1px solid __LINE__; padding: 4px 6px; }
  table.feed td { padding: 4px 6px; border-bottom: 1px solid #E4E7DE; white-space: nowrap; }
  table.feed tr.flag td { background: #F3E1DC; }
  table.feed td.prev { font-family: Consolas, monospace; color: __SOFT__; overflow: hidden; text-overflow: ellipsis; max-width: 260px; }
  .lim ul { margin: 6px 0 0 18px; padding: 0; } .lim li { margin-bottom: 6px; font-size: 13.5px; color: __INK__; }
  .empty { padding: 48px 24px; text-align: center; color: __SOFT__; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""
for _k, _v in {"__PAPER__": PAPER, "__INK__": INK, "__SOFT__": SOFT, "__CARD__": CARD, "__LINE__": LINE, "__GREEN__": GREEN}.items():
    INDEX_HTML = INDEX_HTML.replace(_k, _v)


def build_app(db_path, models_dir, audit_path):
    from dash import Dash, dcc, html, Input, Output

    app = Dash(__name__, title="Live Moderation Monitor")
    app.index_string = INDEX_HTML
    graph_cfg = {"displayModeBar": False}

    def kpi(label, value, sub=""):
        return html.Div([html.Div(value, className="v"), html.Div(label, className="l"),
                         html.Div(sub, className="s")], className="card kpi")

    def render_status(data):
        if data["state"] != "ok":
            return ""
        h = data["health"]
        if data["age_s"] < LIVE_IF_NEWER_THAN_S and not h.get("ended_at"):
            return html.Span("LIVE", className="badge live")
        mins = data["age_s"] / 60
        ago = f"{mins:.0f} min ago" if mins >= 1 else f"{data['age_s']:.0f} s ago"
        why = "collector stopped" if h.get("ended_at") else "no new comments"
        return html.Span(f"PAUSED - {why}, last comment {ago}", className="badge stale")

    def render_kpis(data):
        if data["state"] != "ok":
            return []
        k, h = data["kpis"], data["health"]
        rate = f"{k['rate_per_s']:.1f}/s" if k["rate_per_s"] else "-"
        lag = f"{k['avg_lag']:.1f} s" if k["avg_lag"] is not None else "-"
        return [
            kpi("comments scored", fmt_int(k["n"]), WINDOWS[data["window_key"]]["label"]),
            kpi("flagged", f"{k['flagged_pct']:.1f}%", f"{fmt_int(k['flagged'])} comments, {data['mode_key']} mode"),
            kpi("throughput", rate, "English replies reaching the model"),
            kpi("average lag", lag, "receipt to scored"),
            kpi("dropped (overload)", fmt_int(h["dropped"]), "this run; oldest dropped first"),
            kpi("deletions honored", fmt_int(h["deletes"]), "removed when the author deleted the post"),
            kpi("reconnects", fmt_int(h["reconnects"]), "this run"),
        ]

    def render_feed(data):
        if data["state"] != "ok":
            return ""
        head = html.Thead(html.Tr([html.Th(c) for c in ["scored at", "category", "confidence", "words", "CAPS", "masked preview"]]))
        body = []
        for r in data["feed"]:
            conf = f"{r['confidence']:.2f}" if r["flagged"] else f"({r['confidence']:.2f})"
            body.append(html.Tr([html.Td(r["time"]), html.Td(r["category"]), html.Td(conf), html.Td(str(r["words"])),
                                 html.Td(f"{r['caps']:.0%}"), html.Td(r["preview"], className="prev")],
                                className="flag" if r["flagged"] else ""))
        return html.Table([head, html.Tbody(body)], className="feed")

    def empty_message(data):
        if data["state"] == "no_db":
            msg = ("No database yet. Start the collector in another terminal:  python .\\tox_phase3_stream.py")
        else:
            msg = "The database exists but has no comments yet. Waiting for the collector ..."
        return html.Div(msg, className="card empty")

    limitations = limitation_lines(models_dir, audit_path)

    def graph_card(title, gid, height=300):
        return html.Div([html.H3(title), dcc.Graph(id=gid, config=graph_cfg, style={"height": f"{height}px"})],
                        className="card")

    body = html.Div([
        html.Div(id="kpis", className="kpis"),
        html.Div([graph_card("Flag rate over time", "fig-ts"), graph_card("Flags by category", "fig-cat")],
                 className="grid2"),
        html.Div([graph_card("How confident is the model?", "fig-hist"),
                  html.Div([html.H3("Live feed (masked)"), html.Div(id="feed")], className="card")],
                 className="grid2b"),
        html.Div(id="empty"),
    ], id="body")

    app.layout = html.Div([
        html.H1("Live Moderation Monitor"),
        html.Div("A neural network scoring English replies on Bluesky as they are posted. Flagged content is never "
                 "displayed: only category, confidence and a masked preview.", className="sub"),
        html.Div([
            html.Div([html.Div("Mode", className="t"),
                      dcc.RadioItems(id="mode", options=[{"label": v["label"], "value": k} for k, v in MODES.items()],
                                     value="strict", inline=True)], className="group"),
            html.Div([html.Div("Time window", className="t"),
                      dcc.RadioItems(id="window", options=[{"label": v["label"], "value": k} for k, v in WINDOWS.items()],
                                     value="15m", inline=True)], className="group"),
            html.Div(id="status"),
        ], className="card controls row", style={"alignItems": "flex-end"}),
        body,
        html.Div([html.H3("Limitations - read before trusting any number above"),
                  html.Ul([html.Li(t) for t in limitations])], className="card lim"),
        dcc.Interval(id="tick", interval=WINDOWS["15m"]["refresh_ms"], n_intervals=0),
    ], className="wrap")

    @app.callback(Output("tick", "interval"), Input("window", "value"))
    def set_refresh(window_key):
        return WINDOWS[window_key]["refresh_ms"]

    @app.callback(
        Output("status", "children"), Output("kpis", "children"), Output("fig-ts", "figure"),
        Output("fig-cat", "figure"), Output("fig-hist", "figure"), Output("feed", "children"),
        Output("empty", "children"),
        Input("tick", "n_intervals"), Input("mode", "value"), Input("window", "value"))
    def refresh(_n, mode_key, window_key):
        data = load_dashboard_data(db_path, window_key, mode_key)
        if data["state"] != "ok":
            blank = {"data": [], "layout": _base_layout()}
            return "", [], blank, blank, blank, "", empty_message(data)
        return (render_status(data), render_kpis(data), fig_timeseries(data), fig_categories(data),
                fig_hist(data), render_feed(data), "")

    return app


# ── --check ────────────────────────────────────────────────
def run_check(db_path):
    """Builds every panel for every window x mode once, and validates the
    figures through plotly, without starting a server."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = None
        print("note: plotly not installed (pip install dash); skipping figure validation")
    if not os.path.exists(db_path):
        print(f"CHECK FAILED: {db_path} not found. Run tox_phase3_stream.py first.")
        return 1
    problems = 0
    for wk in WINDOWS:
        for mk in MODES:
            data = load_dashboard_data(db_path, wk, mk)
            if data["state"] != "ok":
                print(f"  {wk:>4} / {mk:<9} state = {data['state']}")
                problems += data["state"] == "no_db"
                continue
            try:
                figs = [fig_timeseries(data), fig_categories(data), fig_hist(data)]
                json.dumps(figs)
                if go is not None:
                    for f in figs:
                        go.Figure(f)
            except Exception as e:
                print(f"  {wk:>4} / {mk:<9} FIGURE ERROR: {type(e).__name__}: {e}")
                problems += 1
                continue
            k = data["kpis"]
            print(f"  {wk:>4} / {mk:<9} ok: {fmt_int(k['n'])} comments, {k['flagged_pct']:.1f}% flagged, "
                  f"{len(data['ts'])} time buckets, {len(data['feed'])} feed rows")
    print("\nCHECK OK" if not problems else f"\nCHECK FOUND {problems} PROBLEM(S)")
    return 1 if problems else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Live dashboard for the toxicity monitor.")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--audit-file", default="audit_summary.json")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--check", action="store_true", help="build every panel once and exit (no server)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.check:
        return run_check(args.db)
    try:
        app = build_app(args.db, args.models_dir, args.audit_file)
    except ImportError:
        print("The 'dash' package is missing. Run:  pip install dash")
        return 1
    print(f"Dashboard running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
