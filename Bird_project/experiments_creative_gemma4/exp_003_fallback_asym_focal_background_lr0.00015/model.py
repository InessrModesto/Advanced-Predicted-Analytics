"""
Experiment 3 (Creative Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = asymmetric focal loss
    augmentation  = background mix
    optimizer     = Nadam
    schedule      = exponential decay
    initial_lr    = 0.00015
    signature     = 4c14fc765d5a

Generated at 2026-05-19 02:29:10
Rationale:
    Fallback free-form recipe: try a conservative asymmetric focal objective with background blending and Nadam for smoother updates.

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
LEARNING_RATE      = 0.00015
STEPS_PER_EPOCH    = 86
TOTAL_TRAIN_STEPS  = 860
WINNER_KERAS_PATH  = "trained_models/crnn_lstm.keras"


# ── Architecture: warm-start from the winning baseline ──────────────────
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH)


# ── Loss ────────────────────────────────────────────────────────────────
class AsymmetricFocalLoss(keras.losses.Loss):
    def __init__(self, gamma_pos=0.0, gamma_neg=3.0, name="asymmetric_focal"):
        super().__init__(name=name)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        pos_loss = y_true * ops.power(1.0 - y_pred, self.gamma_pos) * ops.log(y_pred)
        neg_loss = (1.0 - y_true) * ops.power(y_pred, self.gamma_neg) * ops.log(1.0 - y_pred)
        return -ops.mean(pos_loss + neg_loss)

def get_loss():
    return AsymmetricFocalLoss()


# ── Optimizer + LR schedule ─────────────────────────────────────────────
def get_optimizer():
    lr = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=STEPS_PER_EPOCH,
        decay_rate=0.85,
        staircase=True,
    )
    return keras.optimizers.Nadam(learning_rate=lr)


def get_schedule_callbacks():
    return []


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
def augment_batch(xs, ys):
    if len(xs) < 2:
        return xs, ys
    bg_weight = float(np.random.uniform(0.03, 0.12))
    idx = np.random.permutation(len(xs))
    xs_m = (1.0 - bg_weight) * xs + bg_weight * xs[idx]
    return xs_m.astype(np.float32), ys
