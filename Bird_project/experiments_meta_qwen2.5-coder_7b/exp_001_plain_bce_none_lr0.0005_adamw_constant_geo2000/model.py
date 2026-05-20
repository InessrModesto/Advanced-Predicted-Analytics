"""
Experiment 1 (Meta Agent)
    starts from   = regular's exp 9 (weighted_auc=0.7509)  (loaded from experiments_regular_qwen2.5-coder_7b/exp_009_plain_bce_specaugment_lr0.001_adamw_exp_decay/model.keras)
    loss          = plain_bce
    augmentation  = none
    optimizer     = adamw
    schedule      = constant
    initial_lr    = 0.0005
    geo_scale_km  = 2000

Generated at 2026-05-19 01:26:26
Rationale:
    (fallback: LLM unavailable or unhelpful — first untried)

Self-contained: build_model(), get_loss(), get_optimizer(),
get_schedule_callbacks(), augment_batch(), get_geo_scale_km().
"""
import numpy as np
import keras
from keras import layers, ops

NUM_CLASSES        = 234
LEARNING_RATE      = 0.0005
STEPS_PER_EPOCH    = 86
TOTAL_TRAIN_STEPS  = 860
WINNER_KERAS_PATH  = "experiments_regular_qwen2.5-coder_7b/exp_009_plain_bce_specaugment_lr0.001_adamw_exp_decay/model.keras"
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
    return keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)


def get_schedule_callbacks():
    return [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)]


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
def augment_batch(xs, ys):
    return xs, ys


# ── Geographic sample weighting ─────────────────────────────────────────
def get_geo_scale_km():
    return GEO_SCALE_KM
