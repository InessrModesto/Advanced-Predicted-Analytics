"""
BirdCLEF+ 2026 — Step C: Architecture Comparison
=================================================

Runs four models through the same training/validation pipeline and produces
a comparison table. All models share:
  - The same training data (focal + soundscape, max_rows=3000 by default)
  - The same validation split (held-out soundscape files)
  - The same loss (BCE), optimizer (Adam, lr=1e-3), and batch size (32)
  - The same input shape (mel-spectrogram, 128 mels x ~311 time frames)

What changes between models is only the architecture itself.

Models compared:
  1. baseline_cnn      — small custom CNN, from scratch (124K params)
  2. bigger_cnn        — deeper/wider custom CNN, from scratch
  3. mobilenetv3_small — pretrained on ImageNet
  4. efficientnet_b0   — pretrained on ImageNet

Output:
  - Console summary with timing & AUC per model
  - comparison_results.csv with the full table

Run:   python compare_models.py
"""

# ─── Backend selection (must come before keras import) ──────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# Reuse the data pipeline from baseline.py — single source of truth
# (this also re-runs Cells 1-6 of baseline.py: builds the table, splits,
#  configures the AudioDataset)
from baseline import (
    cfg,
    LABELS, NUM_CLASSES, LABEL_TO_IDX, SPECIES_TO_TAXON,
    train_rows, val_rows,
    AudioDataset,
    TIME_FRAMES,
)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 1: Model builders — one function per architecture
# ─────────────────────────────────────────────────────────────────────────
INPUT_SHAPE = (cfg.n_mels, TIME_FRAMES, 1)         # custom CNNs (1 channel)
INPUT_SHAPE_3CH = (cfg.n_mels, TIME_FRAMES, 3)     # pretrained models (3 channels)


def build_baseline_cnn() -> keras.Model:
    """Same as the original baseline. ~124K params."""
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_bigger_cnn() -> keras.Model:
    """Wider + deeper version of the baseline. ~700K params.
    Same building blocks (Conv-BN-ReLU-Pool), just more of them.
    Lets us isolate "bigger custom CNN" from "pretrained model"."""
    inputs = keras.Input(shape=INPUT_SHAPE)

    # Block 1
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 2
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 3
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="bigger_cnn")


def _build_pretrained(backbone_fn, model_name: str) -> keras.Model:
    """Generic wrapper for pretrained ImageNet backbones.
    The 1-channel mel-spectrogram is repeated to 3 channels so the
    pretrained weights are usable. The classification head is replaced
    with a Dropout + Dense(NUM_CLASSES, sigmoid)."""
    inputs = keras.Input(shape=INPUT_SHAPE)
    # Repeat the single channel into RGB-like 3 channels.
    # Shape: (H, W, 1) -> (H, W, 3)
    x = layers.Concatenate(axis=-1)([inputs, inputs, inputs])

    backbone = backbone_fn(
        include_top=False,             # drop the 1000-way ImageNet classifier
        weights="imagenet",            # load pretrained weights
        input_shape=INPUT_SHAPE_3CH,
        pooling="avg",                 # global average pool the feature map
    )
    x = backbone(x)

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=model_name)


def build_mobilenetv3_small() -> keras.Model:
    """MobileNetV3-Small with ImageNet weights. ~1.5M params."""
    return _build_pretrained(
        keras.applications.MobileNetV3Small, "mobilenetv3_small"
    )


def build_efficientnet_b0() -> keras.Model:
    """EfficientNet-B0 with ImageNet weights. ~4M params."""
    return _build_pretrained(
        keras.applications.EfficientNetB0, "efficientnet_b0"
    )


MODEL_BUILDERS = {
    "baseline_cnn":      build_baseline_cnn,
    "bigger_cnn":        build_bigger_cnn,
    "mobilenetv3_small": build_mobilenetv3_small,
    "efficientnet_b0":   build_efficientnet_b0,
}


# %% ──────────────────────────────────────────────────────────────────────
# Cell 2: Per-model train + evaluate
# ─────────────────────────────────────────────────────────────────────────
def measure_inference_latency_ms(model: keras.Model, n_warmup: int = 5,
                                 n_runs: int = 30) -> float:
    """Median ms per single 5-second window prediction on CPU."""
    dummy = np.zeros((1,) + INPUT_SHAPE, dtype=np.float32)
    # Warm up — first calls include graph build / kernel cache effects
    for _ in range(n_warmup):
        _ = model.predict(dummy, verbose=0)
    # Real measurements
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def train_and_evaluate(model_name: str, epochs: int = 5) -> dict:
    """Train one model with shared hyperparameters and return its row of
    the comparison table."""
    print(f"\n{'=' * 70}")
    print(f"  TRAINING: {model_name}")
    print(f"{'=' * 70}")

    keras.backend.clear_session()
    keras.utils.set_random_seed(cfg.seed)   # same init seed across models

    model = MODEL_BUILDERS[model_name]()
    n_params = int(model.count_params())
    print(f"  Parameters: {n_params:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc", multi_label=True)],
    )

    train_ds = AudioDataset(train_rows, batch_size=cfg.batch_size, shuffle=True)
    val_ds   = AudioDataset(val_rows,   batch_size=cfg.batch_size, shuffle=False)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1
        ),
    ]

    t_train_start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    train_time_s = time.perf_counter() - t_train_start
    epochs_run   = len(history.history["loss"])

    # Save the trained model so later experiments (e.g. the agent) can use it
    # as a starting point or as a reference for comparison.
    models_dir = Path("trained_models")
    models_dir.mkdir(exist_ok=True)
    weights_path = models_dir / f"{model_name}.keras"
    model.save(weights_path)
    print(f"  Saved trained model to: {weights_path}")

    # Inference latency
    print(f"  Measuring inference latency...")
    latency_ms = measure_inference_latency_ms(model)

    # Validation AUC, per-class and per-taxon (matches baseline.py exactly)
    print(f"  Computing validation AUC...")
    all_preds, all_true = [], []
    for i in range(len(val_ds)):
        xb, yb = val_ds[i]
        all_preds.append(model.predict(xb, verbose=0))
        all_true.append(yb)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    per_class_auc = {}
    for cls_idx, label in enumerate(LABELS):
        if y_true[:, cls_idx].sum() == 0:
            per_class_auc[label] = None
            continue
        try:
            per_class_auc[label] = roc_auc_score(
                y_true[:, cls_idx], y_pred[:, cls_idx]
            )
        except Exception:
            per_class_auc[label] = None

    valid_aucs = [v for v in per_class_auc.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    # Per-taxon mean AUC
    taxon_to_aucs: dict[str, list[float]] = {}
    for label, auc in per_class_auc.items():
        if auc is None:
            continue
        taxon = SPECIES_TO_TAXON.get(label, "Unknown")
        taxon_to_aucs.setdefault(taxon, []).append(auc)
    taxon_aucs = {t: float(np.mean(v)) for t, v in taxon_to_aucs.items()}

    # Total budget for the test set: 12 windows per soundscape × N soundscapes.
    # We don't know N exactly, but Kaggle uses around 700 soundscapes typically.
    # Estimate total inference seconds for 700 × 12 = 8400 windows.
    est_inference_min = (latency_ms * 700 * 12) / 1000 / 60

    result = {
        "model":            model_name,
        "params":           n_params,
        "epochs_run":       epochs_run,
        "train_time_s":     round(train_time_s, 1),
        "latency_ms":       round(latency_ms, 1),
        "est_total_inf_min": round(est_inference_min, 1),
        "macro_auc":        round(macro_auc, 4),
        "n_classes_eval":   len(valid_aucs),
        "aves_auc":         round(taxon_aucs.get("Aves", float("nan")), 4),
        "amphibia_auc":     round(taxon_aucs.get("Amphibia", float("nan")), 4),
        "insecta_auc":      round(taxon_aucs.get("Insecta", float("nan")), 4),
        "mammalia_auc":     round(taxon_aucs.get("Mammalia", float("nan")), 4),
    }

    print(f"\n  Result for {model_name}:")
    print(f"    params={n_params:,}  train={train_time_s:.0f}s  "
          f"latency={latency_ms:.1f}ms")
    print(f"    macro_auc={macro_auc:.4f}  "
          f"(over {len(valid_aucs)} evaluable classes)")
    print(f"    per-taxon: " +
          " / ".join(f"{t}={taxon_aucs.get(t, float('nan')):.3f}"
                     for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]))

    return result


# %% ──────────────────────────────────────────────────────────────────────
# Cell 3: Run all four and assemble the table
# ─────────────────────────────────────────────────────────────────────────
EPOCHS = 5  # same for all models for a fair comparison

results = []
for model_name in MODEL_BUILDERS.keys():
    try:
        results.append(train_and_evaluate(model_name, epochs=EPOCHS))
    except Exception as e:
        print(f"\n  [!] {model_name} failed: {type(e).__name__}: {e}")
        results.append({
            "model": model_name, "params": None, "epochs_run": 0,
            "train_time_s": None, "latency_ms": None, "est_total_inf_min": None,
            "macro_auc": None, "n_classes_eval": 0,
            "aves_auc": None, "amphibia_auc": None,
            "insecta_auc": None, "mammalia_auc": None,
        })

# Build the comparison table
df = pd.DataFrame(results)

# Pretty-print
print("\n" + "=" * 70)
print(" COMPARISON TABLE")
print("=" * 70)
print(df.to_string(index=False))

# Save
out_path = Path("comparison_results.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path.resolve()}")

# Quick narrative
print("\n" + "=" * 70)
print(" QUICK READ")
print("=" * 70)
ranked = df.dropna(subset=["macro_auc"]).sort_values("macro_auc", ascending=False)
if len(ranked) > 0:
    best = ranked.iloc[0]
    print(f"  Best macro AUC: {best['model']} "
          f"({best['macro_auc']:.4f}) with {best['params']:,} params")
    smallest = df.dropna(subset=["params"]).sort_values("params").iloc[0]
    print(f"  Smallest:       {smallest['model']} "
          f"({smallest['params']:,} params, "
          f"{smallest['latency_ms']:.1f}ms/window)")
print(f"\n  All models trained for up to {EPOCHS} epochs with identical")
print(f"  hyperparameters (Adam, lr={cfg.lr}, batch={cfg.batch_size}).")
print(f"  Per-model hyperparameter tuning is deferred to Step E (agent).")
