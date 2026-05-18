"""
Experiment 1 (Regular Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = weighted_bce
    augmentation  = specaugment
    optimizer     = adamw
    schedule      = cosine_decay
    initial_lr    = 0.0005

Generated at 2026-05-18 02:03:35
Rationale:
    Since this is the first experiment, I am selecting a robust, balanced baseline configuration. Using weighted_bce addresses the potential class imbalance inherent in multi-label classification across 234 classes, which is crucial for maximizing macro_auc. Specaugment is a standard and highly effective augmentation for audio spectrograms. The combination of adamw (for better generalization via weight decay), 5e-4 (a stable learning rate), and cosine_decay (a smooth convergence schedule) provides a strong, modern starting point for fine-tuning the model.

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
LEARNING_RATE      = 0.0005
STEPS_PER_EPOCH    = 86
TOTAL_TRAIN_STEPS  = 860
WINNER_KERAS_PATH  = "trained_models/crnn_lstm.keras"


# ── Architecture: warm-start from the winning baseline ──────────────────
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH)


# ── Loss ────────────────────────────────────────────────────────────────
POS_WEIGHTS = np.load("experiments_regular_gemma4/class_pos_weights.npy").astype("float32")

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
    return keras.optimizers.AdamW(learning_rate=keras.optimizers.schedules.CosineDecay(initial_learning_rate=LEARNING_RATE, decay_steps=TOTAL_TRAIN_STEPS, alpha=0.0), weight_decay=1e-4)


def get_schedule_callbacks():
    return []


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
def augment_batch(xs, ys):
    """SpecAugment: zero out random time and frequency bands per sample."""
    xs = xs.copy()
    n, h, w, _ = xs.shape
    for i in range(n):
        # Frequency mask (mel-band stripe)
        f = np.random.randint(0, max(1, h // 8))
        if f > 0:
            f0 = np.random.randint(0, max(1, h - f))
            xs[i, f0:f0 + f, :, :] = 0.0
        # Time mask
        t = np.random.randint(0, max(1, w // 8))
        if t > 0:
            t0 = np.random.randint(0, max(1, w - t))
            xs[i, :, t0:t0 + t, :] = 0.0
    return xs, ys
