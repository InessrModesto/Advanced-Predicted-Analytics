"""
Experiment 6 (EDA Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = weighted_bce
    augmentation  = background_mix
    optimizer     = adam
    schedule      = constant
    initial_lr    = 0.0001
    geo_scale_km  = 2000

Generated at 2026-05-17 07:09:54
Rationale:
    This configuration aims to robustly address the dataset's core challenges. Using `weighted_bce` mitigates the severe taxonomic imbalance (Finding 1), which disproportionately affects the macro AUC. Pairing this with `background_mix` directly simulates the soundscape test condition and addresses the spectral domain shift caused by the continuous insect chorus (Finding 3). Furthermore, using `adam` and a `constant` schedule with a lower learning rate (`1e-4`) explores a different optimization path for fine-tuning, while the 2000 km geographic scale provides a moderate weighting, ensuring that some in-domain Pantanal data contributes without overly biasing the model.

Self-contained: build_model(), get_loss(), get_optimizer(),
get_schedule_callbacks(), augment_batch(), get_geo_scale_km().
"""
import numpy as np
import keras
from keras import layers, ops

NUM_CLASSES        = 234
LEARNING_RATE      = 0.0001
STEPS_PER_EPOCH    = 86
TOTAL_TRAIN_STEPS  = 860
WINNER_KERAS_PATH  = "trained_models/crnn_lstm.keras"
GEO_SCALE_KM       = 2000     # None or a float in km


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
    return keras.optimizers.Adam(learning_rate=LEARNING_RATE)


def get_schedule_callbacks():
    return [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)]


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
def augment_batch(xs, ys):
    """Add a low-volume second sample as background. Simulates the
    Pantanal soundscape condition where target calls overlap with a
    continuous chorus (EDA finding 3)."""
    if len(xs) < 2:
        return xs, ys
    bg_weight = float(np.random.uniform(0.05, 0.20))
    idx = np.random.permutation(len(xs))
    xs_m = (1.0 - bg_weight) * xs + bg_weight * xs[idx]
    return xs_m.astype(np.float32), ys


# ── Geographic sample weighting ─────────────────────────────────────────
def get_geo_scale_km():
    """Return the chosen geographic scale (km) or None for no weighting."""
    return GEO_SCALE_KM
