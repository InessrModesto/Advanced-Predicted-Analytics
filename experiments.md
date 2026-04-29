
# Experiment Log

We use this table to track every model experiment.

| Date | Author | Model | Data | Settings | Local metric | Kaggle score | Notes |
|---|---|---|---|---|---|---|---|
| 2026-04-28 | Ksenia | SmallCNN baseline | 512 fast-dev examples from train_audio + train_soundscapes | 1 epoch, batch size 16, device=mps | val_loss=0.0360 | - | Technical pipeline check. The full preprocessing and training pipeline runs locally. |

## How to Record an Experiment

- `Model`: model architecture, for example `SmallCNN`, `EfficientNet-B0`, `ConvNeXt`.
- `Data`: which data was used, for example `train_audio`, `train_soundscapes`, balanced sampling, augmentations.
- `Settings`: epochs, batch size, learning rate, device.
- `Local metric`: validation metric.
- `Kaggle score`: score after submission.
- `Notes`: what changed and what we observed.

