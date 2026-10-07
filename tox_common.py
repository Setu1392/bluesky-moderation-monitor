"""
=============================================================
Real-Time Toxic Content Moderation
Shared constants and text cleaning (imported by every phase)
=============================================================
Author : Setu Patel

Why this file exists: the SAME cleaning function must run on the
training data (Phase 1/2) and on live Reddit comments (Phase 3/4).
If each script carried its own copy, a small edit to one would
silently make the model see differently-shaped text in production
than it saw in training -- train/serve skew -- and nothing would
crash to warn you. One shared function means one place to change.

Cleaning is deliberately light. Keras' TextVectorization layer
(Phase 2) already lowercases and strips punctuation inside the
model itself, so this function only normalizes the things that
would otherwise explode the vocabulary with one-off junk tokens:
URLs, IP addresses, HTML entities, and stray whitespace.
=============================================================
"""

import html
import re

# The six multi-label targets, in a fixed order. The model's output
# layer and every results table in later phases follow this order.
LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_UPPER_RE = re.compile(r"[A-Z]")
_LETTER_RE = re.compile(r"[A-Za-z]")
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_WS_RE = re.compile(r"\s+")


def clean_text(text):
    """Return a lightly normalized copy of `text`.

    - Non-strings (None, NaN) become "" rather than raising, so one
      bad row can never crash a live stream.
    - HTML entities are decoded ("&amp;" -> "&"). Reddit's API returns
      comment bodies with these; the Wikipedia training data rarely has
      them, so this is a no-op there.
    - URLs and IPv4 addresses are replaced with fixed placeholder words,
      so every URL looks like the same token to the model.
    - Runs of whitespace (including newlines) collapse to one space.
    """
    if not isinstance(text, str):
        return ""
    cleaned = html.unescape(text)
    cleaned = _URL_RE.sub(" urltoken ", cleaned)
    cleaned = _IP_RE.sub(" iptoken ", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def upper_ratio(text):
    """Fraction of ASCII letters in `text` that are uppercase (0.0 if none).

    A cheap "shouting" signal. Phase 1 found that mostly-CAPS comments are
    flagged ~6.5x as often as normal-case ones, but Keras' TextVectorization
    lowercases everything, so the model would never see that signal unless it
    is passed in separately. Training (Phase 2) and live scoring (Phases 3-4)
    both call THIS function on the cleaned text, so the model sees the
    feature computed identically in both places.
    """
    if not isinstance(text, str) or not text:
        return 0.0
    letters = len(_LETTER_RE.findall(text))
    if letters == 0:
        return 0.0
    return len(_UPPER_RE.findall(text)) / letters
