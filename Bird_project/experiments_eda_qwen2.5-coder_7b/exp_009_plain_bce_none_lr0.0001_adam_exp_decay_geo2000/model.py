"""
Experiment 9 (EDA Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = plain_bce
    augmentation  = none
    optimizer     = adam
    schedule      = exp_decay
    initial_lr    = 0.0001
    geo_scale_km  = 2000

Generated at 2026-05-18 04:01:20
Rationale:
    (fallback: LLM unavailable or unhelpful — picked the next untried EDA-motivated combination)

Self-contained: build_model(), get_loss(), get_optimizer(),
get_schedule_callbacks(), augment_batch(), get_geo_scale_km().
"""
import numpy as np
import keras
from keras import layers, ops

NUM_CLASSES        = 234
LEARNING_RATE      = 0.0001
STEPS_PER_EPOCH    = 86
TOTAL_TRAIN_STEPS  = 860
WINNER_KERAS_PATH  = "trained_models/crnn_lstm.keras"
GEO_SCALE_KM       = 2000     # None or a float in km


# ── Architecture: warm-start from the winning baseline ──────────────────
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH)


# ── Loss ────────────────────────────────────────────────────────────────
# Plain BCE — no special class needed.

def get_loss():
    return keras.losses.BinaryCrossentropy()


# ── Optimizer + LR schedule ─────────────────────────────────────────────
def get_optimizer():
    return keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=STEPS_PER_EPOCH, decay_rate=0.9, staircase=True))


def get_schedule_callbacks():
    return []


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
def augment_batch(xs, ys):
    return xs, ys


# ── Geographic sample weighting ─────────────────────────────────────────
def get_geo_scale_km():
    """Return the chosen geographic scale (km) or None for no weighting."""
    return GEO_SCALE_KM
