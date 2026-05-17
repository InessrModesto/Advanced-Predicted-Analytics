"""
Experiment 5 (EDA Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = weighted_bce
    augmentation  = mixup
    optimizer     = adamw
    schedule      = cosine_decay
    initial_lr    = 0.001
    geo_scale_km  = 300

Generated at 2026-05-17 06:10:21
Rationale:
    This configuration aims to maximize performance by addressing the three most critical dataset challenges. We use `weighted_bce` to directly mitigate the severe class imbalance (Finding 1) which disproportionately affects the macro AUC metric. We use `mixup` to synthesize multi-label examples, directly addressing the shift from monophonic focal data to complex soundscapes (Finding 4). Finally, setting `geo=300` provides an aggressive weighting toward the Pantanal wetlands (Finding 5), which is crucial given that the test set is highly localized while the training data is global.

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
WINNER_KERAS_PATH  = "trained_models/crnn_lstm.keras"
GEO_SCALE_KM       = 300     # None or a float in km


# ── Architecture: warm-start from the winning baseline ──────────────────
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH)


# ── Loss ────────────────────────────────────────────────────────────────
POS_WEIGHTS = np.load("experiments_eda_gemma4/class_pos_weights.npy").astype("float32")

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
    """Mixup: linearly combine pairs of examples within the batch."""
    if len(xs) < 2:
        return xs, ys
    alpha = 0.4
    lam = float(np.random.beta(alpha, alpha))
    idx = np.random.permutation(len(xs))
    xs_m = lam * xs + (1.0 - lam) * xs[idx]
    ys_m = lam * ys + (1.0 - lam) * ys[idx]
    return xs_m.astype(np.float32), ys_m.astype(np.float32)


# ── Geographic sample weighting ─────────────────────────────────────────
def get_geo_scale_km():
    """Return the chosen geographic scale (km) or None for no weighting."""
    return GEO_SCALE_KM
