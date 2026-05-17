"""
Experiment 2 (EDA Agent)
    baseline      = crnn_lstm  (loaded from trained_models/crnn_lstm.keras)
    loss          = focal_g2.0
    augmentation  = mixup
    optimizer     = sgd_momentum
    schedule      = exp_decay
    initial_lr    = 0.0001
    geo_scale_km  = 1000

Generated at 2026-05-17 02:35:36
Rationale:
    This configuration addresses the long-tail problem within Aves by using focal_g2.0, which is designed to focus on hard-to-classify, rare species (Finding 2). We switch to mixup augmentation to synthesize multi-label examples, directly simulating the high density of simultaneous species observed in the test soundscapes (Finding 4). Finally, we use a moderate geographic weighting (1000km) to aggressively prioritize the Pantanal data while allowing some global context, addressing the geographic domain shift (Finding 5).

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
GEO_SCALE_KM       = 1000     # None or a float in km


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
    return keras.optimizers.SGD(learning_rate=keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=STEPS_PER_EPOCH, decay_rate=0.9, staircase=True), momentum=0.9)


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
