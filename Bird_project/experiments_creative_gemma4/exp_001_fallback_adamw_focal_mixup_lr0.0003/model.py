"""
Experiment 1 (Creative Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = focal loss gamma 1.5
    augmentation  = gentle mixup plus small noise
    optimizer     = AdamW weight_decay 5e-5
    schedule      = cosine decay
    initial_lr    = 0.0003
    signature     = 1734bcda6b33

Generated at 2026-05-19 01:09:17
Rationale:
    Fallback free-form recipe: moderate focal loss, gentle mixed augmentation, AdamW regularization, and smooth decay.

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
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=1.5, name="focal_loss_g1_5"):
        super().__init__(name=name)
        self.gamma = gamma
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        return ops.mean(-ops.power(1.0 - pt, self.gamma) * ops.log(pt))

def get_loss():
    return FocalLoss()


# ── Optimizer + LR schedule ─────────────────────────────────────────────
def get_optimizer():
    lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=TOTAL_TRAIN_STEPS,
        alpha=0.05,
    )
    return keras.optimizers.AdamW(learning_rate=lr, weight_decay=5e-5)


def get_schedule_callbacks():
    return []


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
def augment_batch(xs, ys):
    if len(xs) < 2:
        return xs, ys
    lam = float(np.random.uniform(0.75, 0.95))
    idx = np.random.permutation(len(xs))
    xs_m = lam * xs + (1.0 - lam) * xs[idx]
    ys_m = lam * ys + (1.0 - lam) * ys[idx]
    noise = np.random.normal(0.0, 0.01, size=xs.shape).astype(np.float32)
    return (xs_m + noise).astype(np.float32), ys_m.astype(np.float32)
