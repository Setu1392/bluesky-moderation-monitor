"""
=============================================================
Real-Time Toxic Content Moderation
Scoring helpers (shared by training and the live pipeline)
=============================================================
Author : Setu Patel

Every place that turns raw comment text into model input -- the
training script (Phase 2), the live Reddit scorer (Phase 3) and the
dashboard (Phase 4) -- goes through `make_inputs()` below. One code
path means the model sees identically-prepared data in training and in
production (see tox_common.py for why that matters).

Only the tf.data / Keras helpers touch TensorFlow, and they import it
lazily, so `make_inputs()` and the threshold helpers stay importable
and testable without TensorFlow installed.
=============================================================
"""

import json
import os

import numpy as np

from tox_common import LABEL_COLS, clean_text, upper_ratio

# Names of the model's two inputs. The dict keys passed to the model
# must match these exactly.
TEXT_KEY = "text"
CAPS_KEY = "caps"

MODEL_FILE = "tox_model.keras"
THRESHOLDS_FILE = "thresholds.json"


def make_inputs(cleaned_texts):
    """Turn already-cleaned strings into the model's two input arrays.

    Returns {"text": (N, 1) object array of strings,
             "caps": (N, 1) float32 array of uppercase ratios}.
    The (N, 1) shapes are what Keras' string Input(shape=(1,)) expects.
    """
    cleaned = [t if isinstance(t, str) else "" for t in cleaned_texts]
    text = np.array(cleaned, dtype=object).reshape(-1, 1)
    caps = np.array([upper_ratio(t) for t in cleaned], dtype=np.float32).reshape(-1, 1)
    return {TEXT_KEY: text, CAPS_KEY: caps}


def make_dataset(inputs, labels=None, batch_size=256, shuffle=False, seed=0):
    """Wrap input arrays (and optionally labels) in a batched tf.data pipeline."""
    import tensorflow as tf

    data = inputs if labels is None else (inputs, labels)
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(inputs[TEXT_KEY]), 50_000), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_probs(model, inputs, batch_size=256):
    """Score prepared inputs; returns an (N, 6) float array of label scores."""
    ds = make_dataset(inputs, labels=None, batch_size=batch_size)
    return np.asarray(model.predict(ds, verbose=0), dtype=np.float32)


def score_texts(model, texts):
    """Convenience: raw comment strings in, (N, 6) label scores out."""
    cleaned = [clean_text(t) for t in texts]
    return predict_probs(model, make_inputs(cleaned))


def threshold_vector(thresholds, mode="f1"):
    """Per-label thresholds as an array in LABEL_COLS order.

    mode="f1"             -> thresholds that maximize F1 on validation data
    mode="high_precision" -> thresholds hitting the precision target; a label
                             that could not reach it gets +inf, i.e. it never
                             flags in this stricter mode.
    """
    key = "f1_threshold" if mode == "f1" else "high_precision_threshold"
    vec = []
    for label in LABEL_COLS:
        t = thresholds[key][label]
        vec.append(np.inf if t is None else float(t))
    return np.array(vec, dtype=np.float32)


def load_artifacts(model_dir="models"):
    """Load the trained model and its tuned thresholds from `model_dir`."""
    import keras

    with open(os.path.join(model_dir, THRESHOLDS_FILE)) as f:
        thresholds = json.load(f)
    model = keras.models.load_model(os.path.join(model_dir, thresholds["model_file"]))
    return model, thresholds
