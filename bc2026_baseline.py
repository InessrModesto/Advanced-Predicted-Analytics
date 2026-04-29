"""
Kaggle baseline for the Pantanal bioacoustics competition.

The script is intentionally self-contained:
- trains a compact CNN on 5-second log-mel spectrograms;
- uses both train_audio and labeled train_soundscapes segments;
- writes submission.csv in Kaggle-compatible format.
"""

from __future__ import annotations

import ast
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class CFG:
    seed: int = 42
    sample_rate: int = 32_000
    duration: float = 5.0
    n_fft: int = 2048
    hop_length: int = 500
    n_mels: int = 128
    batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-3
    num_workers: int = 2
    max_train_rows_fast_dev: int = 512
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_data_dir() -> Path:
    env = os.environ.get("BC2026_DATA_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/kaggle/input/birdclef-2026"),
            Path("/kaggle/input/birdclef-2026-data"),
            Path("/Users/kseniadragun/Desktop/Predictive_Analytics/birds /birdclef-2026"),
            Path("/kaggle/input"),
            Path.cwd(),
            Path.cwd().parent,
        ]
    )
    for candidate in candidates:
        if (candidate / "sample_submission.csv").exists() and (candidate / "taxonomy.csv").exists():
            return candidate
        if candidate.exists():
            for child in candidate.iterdir():
                if child.is_dir() and (child / "sample_submission.csv").exists():
                    return child
    raise FileNotFoundError(
        "Could not find dataset. Set BC2026_DATA_DIR to the folder containing train.csv."
    )


def parse_secondary_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (SyntaxError, ValueError):
        pass
    return []


def parse_time_to_seconds(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value)
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds


def read_audio_segment(
    path: Path,
    sample_rate: int,
    duration: float,
    offset: float | None = None,
    random_crop: bool = False,
) -> np.ndarray:
    info = sf.info(str(path))
    target_len = int(sample_rate * duration)
    total_frames = int(info.frames)

    if info.samplerate != sample_rate:
        # Competition files are already 32 kHz, so this is a defensive guard.
        raise ValueError(f"Unexpected sample rate {info.samplerate} in {path}")

    if offset is None:
        if random_crop and total_frames > target_len:
            start = random.randint(0, total_frames - target_len)
        else:
            start = max(0, (total_frames - target_len) // 2)
    else:
        start = max(0, int(offset * sample_rate))

    audio, _ = sf.read(str(path), start=start, frames=target_len, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len]


class LogMel:
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.window = torch.hann_window(cfg.n_fft)
        self.mel_fb = self._build_mel_filterbank()

    def _hz_to_mel(self, hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> torch.Tensor:
        n_freqs = self.cfg.n_fft // 2 + 1
        f_min = torch.tensor(20.0)
        f_max = torch.tensor(float(self.cfg.sample_rate // 2))
        mel_points = torch.linspace(
            self._hz_to_mel(f_min), self._hz_to_mel(f_max), self.cfg.n_mels + 2
        )
        hz_points = self._mel_to_hz(mel_points)
        bins = torch.floor((self.cfg.n_fft + 1) * hz_points / self.cfg.sample_rate).long()
        fb = torch.zeros(self.cfg.n_mels, n_freqs)
        for i in range(1, self.cfg.n_mels + 1):
            left, center, right = bins[i - 1].item(), bins[i].item(), bins[i + 1].item()
            if center > left:
                fb[i - 1, left:center] = torch.linspace(0, 1, center - left)
            if right > center:
                fb[i - 1, center:right] = torch.linspace(1, 0, right - center)
        return fb

    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(audio)
        spec = torch.stft(
            x,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            window=self.window,
            return_complex=True,
        ).abs().pow(2)
        mel = self.mel_fb @ spec
        mel = torch.log1p(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel.unsqueeze(0)


class AudioDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, label_to_idx: dict[str, int], cfg: CFG, train: bool):
        self.rows = rows.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.cfg = cfg
        self.train = train
        self.transform = LogMel(cfg)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[idx]
        offset = None if pd.isna(row["start"]) else float(row["start"])
        audio = read_audio_segment(
            Path(row["path"]),
            sample_rate=self.cfg.sample_rate,
            duration=self.cfg.duration,
            offset=offset,
            random_crop=self.train and pd.isna(row["start"]),
        )
        x = self.transform(audio)
        y = torch.zeros(len(self.label_to_idx), dtype=torch.float32)
        for label in row["labels"]:
            if label in self.label_to_idx:
                y[self.label_to_idx[label]] = 1.0
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 192),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(192, n_classes),
        )

    def _block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_training_table(data_dir: Path, labels: list[str]) -> pd.DataFrame:
    rows = []
    train_csv = pd.read_csv(data_dir / "train.csv")
    for row in train_csv.itertuples(index=False):
        label_list = [row.primary_label] + parse_secondary_labels(getattr(row, "secondary_labels", []))
        label_list = [label for label in label_list if label in labels]
        if not label_list:
            continue
        rows.append(
            {
                "path": str(data_dir / "train_audio" / row.filename),
                "start": np.nan,
                "end": np.nan,
                "labels": sorted(set(label_list)),
                "source": "train_audio",
                "primary": row.primary_label,
            }
        )

    labels_csv = data_dir / "train_soundscapes_labels.csv"
    if labels_csv.exists():
        soundscape_labels = pd.read_csv(labels_csv)
        for row in soundscape_labels.itertuples(index=False):
            label_list = [x for x in str(row.primary_label).split(";") if x in labels]
            if not label_list:
                continue
            rows.append(
                {
                    "path": str(data_dir / "train_soundscapes" / row.filename),
                    "start": parse_time_to_seconds(row.start),
                    "end": parse_time_to_seconds(row.end),
                    "labels": sorted(set(label_list)),
                    "source": "train_soundscape",
                    "primary": label_list[0],
                }
            )

    table = pd.DataFrame(rows)
    table = table[table["path"].map(lambda x: Path(x).exists())].reset_index(drop=True)
    return table


def split_train_val(table: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(table), dtype=bool)
    for _, idxs in table.groupby("primary").groups.items():
        idxs = np.array(list(idxs))
        rng.shuffle(idxs)
        n_val = max(1, int(math.ceil(len(idxs) * 0.1))) if len(idxs) > 5 else 0
        mask[idxs[:n_val]] = True
    return table.loc[~mask].reset_index(drop=True), table.loc[mask].reset_index(drop=True)


def train_model(train_rows: pd.DataFrame, val_rows: pd.DataFrame, labels: list[str], cfg: CFG) -> nn.Module:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    train_ds = AudioDataset(train_rows, label_to_idx, cfg, train=True)
    val_ds = AudioDataset(val_rows, label_to_idx, cfg, train=False) if len(val_rows) else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        if val_ds is not None
        else None
    )

    model = SmallCNN(len(labels)).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= max(1, len(train_ds))

        if val_loader is None:
            print(f"epoch={epoch} train_loss={train_loss:.4f}")
            continue

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(cfg.device, non_blocking=True)
                y = y.to(cfg.device, non_blocking=True)
                loss = criterion(model(x), y)
                val_loss += loss.item() * len(x)
        val_loss /= max(1, len(val_ds))
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    return model


def predict_submission(model: nn.Module, data_dir: Path, labels: list[str], cfg: CFG) -> pd.DataFrame:
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    transform = LogMel(cfg)
    model.eval()

    predictions = np.zeros((len(sample), len(labels)), dtype=np.float32)
    test_dir = data_dir / "test_soundscapes"
    cache: dict[tuple[str, int], np.ndarray] = {}

    with torch.no_grad():
        for i, row_id in enumerate(sample["row_id"].tolist()):
            file_stem, end_time_str = row_id.rsplit("_", 1)
            end_time = int(end_time_str)
            key = (file_stem, end_time)
            if key not in cache:
                path = test_dir / f"{file_stem}.ogg"
                audio = read_audio_segment(
                    path,
                    sample_rate=cfg.sample_rate,
                    duration=cfg.duration,
                    offset=max(0, end_time - 5),
                    random_crop=False,
                )
                cache[key] = audio
            x = transform(cache[key]).unsqueeze(0).to(cfg.device)
            prob = torch.sigmoid(model(x)).cpu().numpy()[0]
            predictions[i] = prob

    sub = sample.copy()
    sub.loc[:, labels] = predictions
    return sub


def main() -> None:
    cfg = CFG()
    if os.environ.get("BC2026_FAST_DEV") == "1":
        cfg.epochs = 1
        cfg.batch_size = 16

    seed_everything(cfg.seed)
    data_dir = find_data_dir()
    print(f"data_dir={data_dir}")
    print(f"device={cfg.device}")

    sample = pd.read_csv(data_dir / "sample_submission.csv")
    labels = [c for c in sample.columns if c != "row_id"]
    print(f"n_classes={len(labels)}")

    train_table = build_training_table(data_dir, labels)
    if os.environ.get("BC2026_FAST_DEV") == "1":
        train_table = train_table.sample(
            n=min(len(train_table), cfg.max_train_rows_fast_dev),
            random_state=cfg.seed,
        ).reset_index(drop=True)
    elif os.environ.get("BC2026_MAX_ROWS"):
        max_rows = int(os.environ["BC2026_MAX_ROWS"])
        train_table = train_table.sample(
            n=min(len(train_table), max_rows),
            random_state=cfg.seed,
        ).reset_index(drop=True)

    print(train_table["source"].value_counts(dropna=False))
    print(f"train_examples={len(train_table)}")

    train_rows, val_rows = split_train_val(train_table, cfg.seed)
    print(f"train={len(train_rows)} val={len(val_rows)}")

    model = train_model(train_rows, val_rows, labels, cfg)

    if (data_dir / "test_soundscapes").exists() and len(list((data_dir / "test_soundscapes").glob("*.ogg"))):
        submission = predict_submission(model, data_dir, labels, cfg)
    else:
        print("No test soundscapes found. Writing neutral sample-shaped submission.")
        submission = sample.copy()
        submission.loc[:, labels] = 0.01

    out_path = Path("/kaggle/working/submission.csv") if Path("/kaggle").exists() else Path("submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
