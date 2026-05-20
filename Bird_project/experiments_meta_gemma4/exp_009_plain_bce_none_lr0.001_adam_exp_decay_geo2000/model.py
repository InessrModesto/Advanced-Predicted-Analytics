"""
Experiment 9 (Meta Agent)
    starts from   = regular's exp 8 (weighted_auc=0.7529)  (loaded from experiments_regular_gemma4/exp_008_weighted_bce_specaugment_lr0.0005_adamw_constant/model.keras)
    loss          = plain_bce
    augmentation  = none
    optimizer     = adam
    schedule      = exp_decay
    initial_lr    = 0.001
    geo_scale_km  = 2000

Generated at 2026-05-20 01:02:53
Rationale:
    (fallback: LLM unavailable or unhelpful — first untried)

Self-contained: build_model(), get_loss(), get_optimizer(),
get_schedule_callbacks(), augment_batch(), get_geo_scale_km().
"""
import numpy as np
import keras
from keras import layers, ops

NUM_CLASSES        = 234
LEARNING_RATE      = 0.001
STEPS_PER_EPOCH    = 86
TOTAL_TRAIN_STEPS  = 860
WINNER_KERAS_PATH  = "experiments_regular_gemma4/exp_008_weighted_bce_specaugment_lr0.0005_adamw_constant/model.keras"
GEO_SCALE_KM       = 2000


# ── Warm-start from the GLOBAL WINNER (not from the CRNN baseline) ──────
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH, compile=False)


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
    return GEO_SCALE_KM
