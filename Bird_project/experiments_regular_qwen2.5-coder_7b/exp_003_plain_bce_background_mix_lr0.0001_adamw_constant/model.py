"""
Experiment 3 (Regular Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = plain_bce
    augmentation  = background_mix
    optimizer     = adamw
    schedule      = constant
    initial_lr    = 0.0001

Generated at 2026-05-17 01:09:15
Rationale:
    This combination aims to explore a different augmentation technique (background mix) with a lower learning rate and the AdamW optimizer, which can help in fine-tuning the model without overfitting. The constant learning rate schedule allows for more stable convergence during training.

Self-contained: build_model(), get_loss(), get_optimizer(),
get_schedule_callbacks(), augment_batch(). The architecture lives inside
the .keras file referenced by WINNER_KERAS_PATH — Keras deserialises it
from the embedded config.json. At submission time, the winning
experiment's model.py + model.keras are the deliverable.
"""
import numpy as np
import keras
from keras import layers, ops

NUM_CLASSES        = 234
LEARNING_RATE      = 0.0001
STEPS_PER_EPOCH    = 86
TOTAL_TRAIN_STEPS  = 860
WINNER_KERAS_PATH  = "trained_models/crnn_lstm.keras"


# ── Architecture: warm-start from the winning baseline ──────────────────
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH)


# ── Loss ────────────────────────────────────────────────────────────────
# Plain binary cross-entropy — no special class needed.

def get_loss():
    return keras.losses.BinaryCrossentropy()


# ── Optimizer + LR schedule ─────────────────────────────────────────────
def get_optimizer():
    return keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)


def get_schedule_callbacks():
    return [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)]


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
def augment_batch(xs, ys):
    """Mix each input with a random other sample in the batch at low volume.
    Labels are unchanged (we treat the second sample as 'background noise',
    not as a second positive). This roughly simulates the soundscape
    condition where target calls overlap with continuous chorus."""
    if len(xs) < 2:
        return xs, ys
    bg_weight = float(np.random.uniform(0.05, 0.20))
    idx = np.random.permutation(len(xs))
    xs_m = (1.0 - bg_weight) * xs + bg_weight * xs[idx]
    return xs_m.astype(np.float32), ys
