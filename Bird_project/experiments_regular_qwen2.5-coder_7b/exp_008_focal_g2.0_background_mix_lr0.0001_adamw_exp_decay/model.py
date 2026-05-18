"""
Experiment 8 (Regular Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = focal_g2.0
    augmentation  = background_mix
    optimizer     = adamw
    schedule      = exp_decay
    initial_lr    = 0.0001

Generated at 2026-05-17 02:15:34
Rationale:
    The combination of focal loss with a lower learning rate and exponential decay scheduling has shown promise in previous experiments, while background mix augmentation can help improve generalization.

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
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        log_pt = ops.log(pt)
        loss = -ops.power(1.0 - pt, self.gamma) * log_pt
        return ops.mean(loss)


def get_loss():
    return FocalLoss()


# ── Optimizer + LR schedule ─────────────────────────────────────────────
def get_optimizer():
    return keras.optimizers.AdamW(learning_rate=keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=STEPS_PER_EPOCH, decay_rate=0.9, staircase=True), weight_decay=1e-4)


def get_schedule_callbacks():
    return []


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
