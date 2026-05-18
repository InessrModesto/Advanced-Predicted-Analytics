"""
Experiment 8 (Creative Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = binary cross-entropy with label smoothing
    augmentation  = time shift
    optimizer     = Adam
    schedule      = cosine decay
    initial_lr    = 0.0003
    signature     = 00eefbecf81a

Generated at 2026-05-17 14:00:14
Rationale:
    Fallback free-form recipe: mild label smoothing and time shifts probe whether the baseline benefits from calibration and invariance. Deterministic fallback variant with LR scale 0.60.

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
LEARNING_RATE      = 0.0003
STEPS_PER_EPOCH    = 86
TOTAL_TRAIN_STEPS  = 860
WINNER_KERAS_PATH  = "trained_models/crnn_lstm.keras"


# ── Architecture: warm-start from the winning baseline ──────────────────
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH)


# ── Loss ────────────────────────────────────────────────────────────────
# Built-in BCE with label smoothing.

def get_loss():
    return keras.losses.BinaryCrossentropy(label_smoothing=0.02)


# ── Optimizer + LR schedule ─────────────────────────────────────────────
def get_optimizer():
    lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=TOTAL_TRAIN_STEPS,
        alpha=0.1,
    )
    return keras.optimizers.Adam(learning_rate=lr)


def get_schedule_callbacks():
    return []


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
def augment_batch(xs, ys):
    xs = xs.copy()
    _, _, w, _ = xs.shape
    max_shift = max(1, w // 12)
    for i in range(len(xs)):
        shift = int(np.random.randint(-max_shift, max_shift + 1))
        xs[i] = np.roll(xs[i], shift=shift, axis=1)
    return xs, ys
