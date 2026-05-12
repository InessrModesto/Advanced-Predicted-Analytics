"""
BirdCLEF+ 2026 — Baseline_v1 (Keras 3)
====================================

Goal: a single end-to-end pipeline that we can trust.
- Loads focal recordings (train_audio) AND labeled soundscape segments
  (train_soundscapes_labels.csv).
- Trains a small CNN on log-mel spectrograms of 5-second windows.
- Reports per-class and per-taxon AUC on a held-out validation set.
- No augmentation, no fancy models. We add those later as separate steps.

Strict rules respected (from EDA findings):
- Multi-label, 234 sigmoid outputs, BCE loss.
- 5-second windows at 32 kHz mono.
- No spectrogram flips.
- No filtering on `rating`.
- Validation reports per-taxon AUC, not just macro.

Run:   python baseline_v1.py
"""

# %% ──────────────────────────────────────────────────────────────────────
# Cell 1: Imports and config
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

# Keras 3 backend selection. Set BEFORE importing keras.
# Default to torch because it installed cleanly on Windows.
# If you've successfully installed tensorflow, you can flip this to "tensorflow".
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import ast
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import keras
from keras import layers
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ─── Config ──────────────────────────────────────────────────────────────
@dataclass
class CFG:
    # Paths
    data_dir: Path = Path(".")           # project root contains the CSVs and audio folders

    # Audio
    sample_rate: int = 32_000
    duration_sec: float = 5.0            # eval unit
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128

    # Training
    batch_size: int = 32
    epochs: int = 3
    lr: float = 1e-3
    seed: int = 42

    # Debugging knob — set to a small number (e.g. 3000) for fast iteration,
    # then set to None for a real full-data run once the pipeline is trusted.
    max_rows: int | None = 3000


cfg = CFG()
random.seed(cfg.seed)
np.random.seed(cfg.seed)
keras.utils.set_random_seed(cfg.seed)

DATA_DIR          = cfg.data_dir
TRAIN_CSV         = DATA_DIR / "train.csv"
TAXONOMY_CSV      = DATA_DIR / "taxonomy.csv"
SAMPLE_SUB_CSV    = DATA_DIR / "sample_submission.csv"
TRAIN_AUDIO_DIR   = DATA_DIR / "train_audio"
SOUNDSCAPE_DIR    = DATA_DIR / "train_soundscapes"
SOUNDSCAPE_LABELS = DATA_DIR / "train_soundscapes_labels.csv"


# %% ──────────────────────────────────────────────────────────────────────
# Cell 2: Label space — derived from sample_submission.csv (the official 234)
# ─────────────────────────────────────────────────────────────────────────
print("=== Loading label space ===")
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
LABELS = [c for c in sample_sub.columns if c != "row_id"]
NUM_CLASSES = len(LABELS)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
print(f"Label space: {NUM_CLASSES} classes (from sample_submission.csv)")

# Per-class taxonomy lookup (Aves, Amphibia, etc.) — used for per-taxon AUC.
tax_df = pd.read_csv(TAXONOMY_CSV)
SPECIES_TO_TAXON = dict(zip(tax_df["primary_label"].astype(str), tax_df["class_name"]))
IDX_TO_TAXON = {
    i: SPECIES_TO_TAXON.get(label, "Unknown") for i, label in enumerate(LABELS)
}
print(f"Taxon distribution in label space: "
      f"{pd.Series(list(IDX_TO_TAXON.values())).value_counts().to_dict()}")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 3: Build the training table (focal + soundscape segments)
# Each row = one trainable 5-second example.
# - Focal rows: random crop a 5s window from the focal clip at training time.
# - Soundscape rows: load the exact 5s window specified by (start, end).
# ─────────────────────────────────────────────────────────────────────────
def parse_secondary(value: object) -> list[str]:
    """train.csv stores secondary_labels as a stringified list, e.g. "['x','y']" ."""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except (SyntaxError, ValueError):
        return []


def parse_time_to_seconds(value: object) -> float:
    """Soundscape labels store start/end as 'HH:MM:SS'."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value)
    if ":" not in text:
        return float(text)
    seconds = 0.0
    for part in [float(p) for p in text.split(":")]:
        seconds = seconds * 60.0 + part
    return seconds


def build_training_table() -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      path     — absolute path to the audio file
      start    — start offset in seconds (NaN means 'random crop at training')
      labels   — list of species codes present in this example
      source   — 'focal' or 'soundscape'
      primary  — primary species (used for stratified split)
    """
    rows = []

    # 1. Focal recordings
    train_csv = pd.read_csv(TRAIN_CSV)
    for r in train_csv.itertuples(index=False):
        labs = [str(r.primary_label)] + parse_secondary(getattr(r, "secondary_labels", []))
        labs = [l for l in labs if l in LABEL_TO_IDX]
        if not labs:
            continue
        rows.append({
            "path":    str(TRAIN_AUDIO_DIR / r.filename),
            "start":   np.nan,
            "labels":  sorted(set(labs)),
            "source":  "focal",
            "primary": str(r.primary_label),
        })

    # 2. Labeled soundscape segments
    if SOUNDSCAPE_LABELS.exists():
        ss = pd.read_csv(SOUNDSCAPE_LABELS)
        for r in ss.itertuples(index=False):
            # primary_label here is a semicolon-separated string of species codes
            labs = [x for x in str(r.primary_label).split(";") if x in LABEL_TO_IDX]
            if not labs:
                continue
            rows.append({
                "path":    str(SOUNDSCAPE_DIR / r.filename),
                "start":   parse_time_to_seconds(r.start),
                "labels":  sorted(set(labs)),
                "source":  "soundscape",
                "primary": labs[0],
            })

    table = pd.DataFrame(rows)
    # Drop any rows whose audio file doesn't actually exist on disk.
    # This guard prevents the "every clip skipped" bug.
    before = len(table)
    table = table[table["path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    after = len(table)
    print(f"Built training table: {after} rows ({before - after} dropped for missing files)")
    print(f"  source breakdown:\n{table['source'].value_counts().to_string()}")
    return table


print("\n=== Building training table ===")
table = build_training_table()
if cfg.max_rows is not None:
    table = table.sample(n=min(cfg.max_rows, len(table)), random_state=cfg.seed).reset_index(drop=True)
    print(f"  (subsampled to {len(table)} rows for fast iteration)")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 4: Stratified train/val split
# ─────────────────────────────────────────────────────────────────────────
def split_train_val(table: pd.DataFrame, val_frac: float = 0.10, seed: int = 42):
    """
    Hold out `val_frac` of each species. Species with <=5 examples stay entirely
    in train (we cannot validate on classes we have no spare examples of).
    """
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(table), dtype=bool)
    for primary, idxs in table.groupby("primary").groups.items():
        idxs = np.array(list(idxs))
        rng.shuffle(idxs)
        n_val = max(1, int(math.ceil(len(idxs) * val_frac))) if len(idxs) > 5 else 0
        val_mask[idxs[:n_val]] = True
    train = table.loc[~val_mask].reset_index(drop=True)
    val   = table.loc[ val_mask].reset_index(drop=True)
    return train, val


train_rows, val_rows = split_train_val(table, val_frac=0.10, seed=cfg.seed)
print(f"\nSplit: {len(train_rows)} train / {len(val_rows)} val")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 5: Audio loader and mel-spectrogram (pure NumPy, no librosa needed)
# ─────────────────────────────────────────────────────────────────────────
def read_audio_segment(path: Path, sample_rate: int, duration: float,
                       offset: float | None = None,
                       random_crop: bool = False) -> np.ndarray:
    """Load a fixed-length audio segment. Pad with zeros if too short."""
    info = sf.info(str(path))
    target_len = int(sample_rate * duration)
    total = int(info.frames)

    if info.samplerate != sample_rate:
        # We expect 32 kHz across the dataset; raise loudly if not.
        raise ValueError(f"Sample rate mismatch in {path}: {info.samplerate} != {sample_rate}")

    if offset is None:
        if random_crop and total > target_len:
            start = random.randint(0, total - target_len)
        else:
            start = max(0, (total - target_len) // 2)
    else:
        start = max(0, int(offset * sample_rate))

    audio, _ = sf.read(str(path), start=start, frames=target_len,
                       dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo -> mono (shouldn't happen, but safety)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len]


def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Standard slaney-style mel filterbank, shape (n_mels, n_fft//2 + 1)."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    f_min, f_max = 20.0, sr / 2.0
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c > l: fb[i - 1, l:c] = np.linspace(0, 1, c - l, dtype=np.float32)
        if r > c: fb[i - 1, c:r] = np.linspace(1, 0, r - c, dtype=np.float32)
    return fb


MEL_FB   = _build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels)
HANN_WIN = np.hanning(cfg.n_fft).astype(np.float32)


def waveform_to_mel(audio: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram. Output shape: (n_mels, time_frames).
    Vectorized: builds all frames at once via stride tricks, then a single
    batched FFT over all frames. ~30x faster than a Python loop on CPU."""
    n = len(audio)
    n_frames = max(1, 1 + (n - cfg.n_fft) // cfg.hop_length)
    # Pad the tail so the last frame fits cleanly
    needed = (n_frames - 1) * cfg.hop_length + cfg.n_fft
    if n < needed:
        audio = np.pad(audio, (0, needed - n))

    # Build a 2D view of shape (n_frames, n_fft) with NO copy
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, cfg.n_fft),
        strides=(audio.strides[0] * cfg.hop_length, audio.strides[0]),
        writeable=False,
    )
    # Apply window + batched real FFT (n_frames, n_fft//2+1)
    windowed = frames * HANN_WIN
    spec = np.abs(np.fft.rfft(windowed, axis=1)) ** 2   # (n_frames, n_freqs)
    spec = spec.T                                        # (n_freqs, n_frames)

    mel = MEL_FB @ spec
    mel = np.log1p(mel)
    # Per-clip standardization
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.astype(np.float32)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 6: Keras PyDataset — yields (X, y) batches lazily from disk
# ─────────────────────────────────────────────────────────────────────────
class AudioDataset(keras.utils.PyDataset):
    """One sample = (mel_spectrogram, multi-hot label vector)."""

    def __init__(self, rows: pd.DataFrame, batch_size: int, shuffle: bool, **kwargs):
        # Default to multiprocessing across CPU cores. On Windows, Keras handles
        # this safely as long as kwargs don't already specify it.
        kwargs.setdefault("workers", 4)
        kwargs.setdefault("use_multiprocessing", False)  # threads are safer on Windows
        kwargs.setdefault("max_queue_size", 8)
        super().__init__(**kwargs)
        self.rows = rows.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.rows))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.rows) / self.batch_size)

    def __getitem__(self, batch_idx):
        batch_indices = self.indices[batch_idx * self.batch_size:
                                     (batch_idx + 1) * self.batch_size]
        xs, ys = [], []
        for i in batch_indices:
            row = self.rows.iloc[i]
            offset = None if pd.isna(row["start"]) else float(row["start"])
            audio = read_audio_segment(
                Path(row["path"]),
                sample_rate=cfg.sample_rate,
                duration=cfg.duration_sec,
                offset=offset,
                random_crop=(self.shuffle and pd.isna(row["start"])),
            )
            mel = waveform_to_mel(audio)            # (n_mels, time)
            mel = mel[..., np.newaxis]               # (n_mels, time, 1)
            xs.append(mel)

            y = np.zeros(NUM_CLASSES, dtype=np.float32)
            for label in row["labels"]:
                y[LABEL_TO_IDX[label]] = 1.0
            ys.append(y)
        return np.stack(xs), np.stack(ys)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# Quick sanity check on shapes
print("\n=== Shape sanity check ===")
probe_ds = AudioDataset(train_rows.head(4), batch_size=4, shuffle=False)
xb, yb = probe_ds[0]
expected_frames = int(cfg.sample_rate * cfg.duration_sec / cfg.hop_length)
print(f"  X batch shape: {xb.shape}    (expected approx (4, {cfg.n_mels}, {expected_frames}, 1))")
print(f"  y batch shape: {yb.shape}    (expected: (4, {NUM_CLASSES}))")
print(f"  X range:       [{xb.min():.2f}, {xb.max():.2f}]")
TIME_FRAMES = xb.shape[2]


# %% ──────────────────────────────────────────────────────────────────────
# Cell 7: Build the model — small CNN, Functional API, sigmoid output
# Strict rules: ReLU hidden, sigmoid output, BCE loss.
# ─────────────────────────────────────────────────────────────────────────
def build_model(n_classes: int, input_shape: tuple[int, int, int]) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
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
    outputs = layers.Dense(n_classes, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


model = build_model(NUM_CLASSES, input_shape=(cfg.n_mels, TIME_FRAMES, 1))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cfg.lr),
    loss=keras.losses.BinaryCrossentropy(),  # sigmoid output -> plain BCE
    metrics=[keras.metrics.AUC(name="auc", multi_label=True)],
)
print("\n=== Model summary ===")
model.summary()
print(f"Total parameters: {model.count_params():,}")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 8: Train
# ─────────────────────────────────────────────────────────────────────────
print(f"\n=== Training for up to {cfg.epochs} epochs ===")

train_ds = AudioDataset(train_rows, batch_size=cfg.batch_size, shuffle=True)
val_ds   = AudioDataset(val_rows,   batch_size=cfg.batch_size, shuffle=False)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="baseline_best.keras", monitor="val_loss",
        save_best_only=True, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg.epochs,
    callbacks=callbacks,
    verbose=1,
)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 9: Validation — per-class AUC and per-taxon AUC
# This is the metric block every later experiment compares against.
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Computing validation AUC ===")

all_preds, all_true = [], []
for i in range(len(val_ds)):
    xb, yb = val_ds[i]
    pb = model.predict(xb, verbose=0)
    all_preds.append(pb)
    all_true.append(yb)
y_pred = np.concatenate(all_preds, axis=0)
y_true = np.concatenate(all_true, axis=0)
print(f"  validation predictions shape: {y_pred.shape}")

# Per-class AUC, skipping classes with no positive examples in val
# (matches the competition metric, which also skips them).
per_class_auc: dict[str, float | None] = {}
for cls_idx, label in enumerate(LABELS):
    if y_true[:, cls_idx].sum() == 0:
        per_class_auc[label] = None
        continue
    try:
        per_class_auc[label] = roc_auc_score(y_true[:, cls_idx], y_pred[:, cls_idx])
    except Exception:
        per_class_auc[label] = None

valid_aucs = [v for v in per_class_auc.values() if v is not None]
macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

# Per-taxon AUC
print("\n=== Per-taxon AUC ===")
taxon_to_aucs: dict[str, list[float]] = {}
for label, auc in per_class_auc.items():
    if auc is None:
        continue
    taxon = SPECIES_TO_TAXON.get(label, "Unknown")
    taxon_to_aucs.setdefault(taxon, []).append(auc)

for taxon in sorted(taxon_to_aucs.keys()):
    aucs = taxon_to_aucs[taxon]
    print(f"  {taxon:<12} mean AUC = {np.mean(aucs):.3f}   "
          f"(n_classes_evaluated={len(aucs)})")

print(f"\n=== Overall macro AUC: {macro_auc:.4f}  "
      f"(over {len(valid_aucs)} classes with positive validation examples) ===")

# How many classes have ZERO positive examples in val? Worth knowing.
print("\nClasses skipped (no positive validation examples):")
skipped_by_taxon: dict[str, int] = {}
for label, auc in per_class_auc.items():
    if auc is None:
        taxon = SPECIES_TO_TAXON.get(label, "Unknown")
        skipped_by_taxon[taxon] = skipped_by_taxon.get(taxon, 0) + 1
for taxon, n in sorted(skipped_by_taxon.items()):
    print(f"  {taxon:<12} {n} classes skipped")


print("\n=== Pipeline complete ===")
print("If you got here, the pipeline runs end-to-end on the real data.")
print("The macro AUC and per-taxon AUC above are the numbers every later")
print("experiment will be compared against.")
