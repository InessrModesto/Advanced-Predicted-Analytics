"""
Experiment 10 (Creative Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = weighted BCE with clipped positive weights
    augmentation  = moderate SpecAugment
    optimizer     = AdamW weight_decay 1e-4
    schedule      = constant LR with ReduceLROnPlateau
    initial_lr    = 0.0001
    signature     = b69d8f4aacdc

Generated at 2026-05-19 14:07:10
Rationale:
    Fallback free-form recipe: stabilize warm-start fine-tuning with weighted BCE, moderate masking, AdamW, and plateau-based LR cuts. Deterministic fallback variant with LR scale 0.50.

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
POS_WEIGHTS = np.load("experiments_creative_gemma4/class_pos_weights.npy").astype("float32")

class WeightedBCE(keras.losses.Loss):
    def __init__(self, pos_weights, name="weighted_bce"):
        super().__init__(name=name)
        self.pos_weights = ops.convert_to_tensor(pos_weights)
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        per_class = -(self.pos_weights * y_true * ops.log(y_pred)
                      + (1.0 - y_true) * ops.log(1.0 - y_pred))
        return ops.mean(per_class)

def get_loss():
    return WeightedBCE(POS_WEIGHTS)


# ── Optimizer + LR schedule ─────────────────────────────────────────────
def get_optimizer():
    return keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)


def get_schedule_callbacks():
    return [keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=2e-5, verbose=0
    )]


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
def augment_batch(xs, ys):
    xs = xs.copy()
    n, h, w, _ = xs.shape
    for i in range(n):
        f = np.random.randint(0, max(1, h // 10))
        if f > 0:
            f0 = np.random.randint(0, max(1, h - f))
            xs[i, f0:f0 + f, :, :] = 0.0
        t = np.random.randint(0, max(1, w // 10))
        if t > 0:
            t0 = np.random.randint(0, max(1, w - t))
            xs[i, :, t0:t0 + t, :] = 0.0
    return xs, ys
