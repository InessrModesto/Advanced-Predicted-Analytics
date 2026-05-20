"""
Experiment 2 (Meta Agent)
    starts from   = regular's exp 8 (weighted_auc=0.7529)  (loaded from experiments_regular_gemma4/exp_008_weighted_bce_specaugment_lr0.0005_adamw_constant/model.keras)
    loss          = weighted_bce
    augmentation  = specaugment
    optimizer     = adamw
    schedule      = exp_decay
    initial_lr    = 0.001
    geo_scale_km  = 5000

Generated at 2026-05-19 18:54:56
Rationale:
    The global winner and meta-experiment both strongly suggest sticking with 'weighted_bce' and 'specaugment' as the best performing combination for loss and augmentation. We maintain 'adamw' as the optimizer, which is the most frequent in the top 10 runs. To explore a new high-potential area, we are increasing the initial learning rate to 1e-3, which is the standard rate and has not been the primary focus of the top runs. We are testing 'exp_decay' for the schedule, as it is less represented in the top results, and setting the geographic weight to 5000km to test a mild, non-zero weighting that hasn't been explored in the top configurations.

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
WINNER_KERAS_PATH  = "experiments_regular_gemma4/exp_008_weighted_bce_specaugment_lr0.0005_adamw_constant/model.keras"
GEO_SCALE_KM       = 5000


# ── Warm-start from the GLOBAL WINNER (not from the CRNN baseline) ──────
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH, compile=False)


# ── Loss ────────────────────────────────────────────────────────────────
POS_WEIGHTS = np.load("experiments_meta_gemma4/class_pos_weights.npy").astype("float32")

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
    return keras.optimizers.AdamW(learning_rate=keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=STEPS_PER_EPOCH, decay_rate=0.9, staircase=True), weight_decay=1e-4)


def get_schedule_callbacks():
    return []


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
def augment_batch(xs, ys):
    """SpecAugment: random time/frequency masking."""
    xs = xs.copy()
    n, h, w, _ = xs.shape
    for i in range(n):
        f = np.random.randint(0, max(1, h // 8))
        if f > 0:
            f0 = np.random.randint(0, max(1, h - f))
            xs[i, f0:f0 + f, :, :] = 0.0
        t = np.random.randint(0, max(1, w // 8))
        if t > 0:
            t0 = np.random.randint(0, max(1, w - t))
            xs[i, :, t0:t0 + t, :] = 0.0
    return xs, ys


# ── Geographic sample weighting ─────────────────────────────────────────
def get_geo_scale_km():
    return GEO_SCALE_KM
