"""
Experiment 1 (Meta Agent)
    starts from   = regular's exp 8 (weighted_auc=0.7529)  (loaded from experiments_regular_gemma4/exp_008_weighted_bce_specaugment_lr0.0005_adamw_constant/model.keras)
    loss          = weighted_bce
    augmentation  = specaugment
    optimizer     = adamw
    schedule      = cosine_decay
    initial_lr    = 0.0001
    geo_scale_km  = 2000

Generated at 2026-05-19 18:18:40
Rationale:
    The Global Winner utilized the optimal combination of `weighted_bce` and `specaugment`. I am maintaining these strong components while systematically exploring improvements. I am switching the schedule from the GW's `constant` to `cosine_decay`, which is the most frequent schedule in the top 10 and shows strong performance. Furthermore, I am reducing the learning rate from 5e-4 to 1e-4 for a gentler fine-tuning step. Finally, I am introducing a moderate geographic weight (2000km) to test the impact of the geographical weighting, which has not been explored in the top-performing models.

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
WINNER_KERAS_PATH  = "experiments_regular_gemma4/exp_008_weighted_bce_specaugment_lr0.0005_adamw_constant/model.keras"
GEO_SCALE_KM       = 2000


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
    return keras.optimizers.AdamW(learning_rate=keras.optimizers.schedules.CosineDecay(initial_learning_rate=LEARNING_RATE, decay_steps=TOTAL_TRAIN_STEPS, alpha=0.0), weight_decay=1e-4)


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
