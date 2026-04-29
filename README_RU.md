# Kaggle Pantanal Bioacoustics: рабочий план

Цель проекта: предсказывать присутствие 234 видов/сонотипов в каждом 5-секундном сегменте 1-минутных soundscape-аудио.

## Командная работа

Основной файл для GitHub: [README.md](/Users/kseniadragun/Documents/Codex/2026-04-28/overview-the-goal-of-this-competition/README.md).

Важно: датасет, аудио, веса моделей и `submission.csv` не нужно загружать в GitHub. Они добавлены в `.gitignore`. Каждый участник команды скачивает датасет себе локально и задает путь через `BC2026_DATA_DIR`.

## Что уже важно понимать

- `train_audio/` содержит короткие записи отдельных видов. Это основной supervised-сигнал.
- `train_soundscapes/` ближе к тесту по домену: длинные полевые записи с шумом, несколькими видами и реальными условиями.
- `train_soundscapes_labels.csv` особенно ценен: некоторые виды из hidden test могут встречаться только там.
- `sample_submission.csv` задает точный список 234 классов и формат строк.
- Тест считается по 5-секундным окнам: `..._20` означает сегмент `00:15-00:20`.

## Минимальный baseline

Файл [notebooks/bc2026_baseline.py](/Users/kseniadragun/Documents/Codex/2026-04-28/overview-the-goal-of-this-competition/notebooks/bc2026_baseline.py) делает полный цикл:

1. Находит датасет локально или в Kaggle.
2. Собирает обучающие примеры из `train.csv` и `train_soundscapes_labels.csv`.
3. Превращает 5 секунд аудио в log-mel спектрограмму.
4. Обучает компактную CNN с multi-label BCE loss.
5. Создает `submission.csv` для скрытого `test_soundscapes`.

## Как запускать на Kaggle

1. Создай Kaggle Notebook в соревновании.
2. Добавь competition dataset в Inputs.
3. Вставь содержимое `notebooks/bc2026_baseline.py` в notebook или загрузи как script.
4. Включи GPU.
5. Запусти все ячейки/скрипт.
6. Убедись, что появился `/kaggle/working/submission.csv`.

## Следующие улучшения

- Заменить маленькую CNN на EfficientNet/ConvNeXt с pretrained-весами.
- Делать balanced sampling по редким классам.
- Использовать mixup/cutmix на спектрограммах.
- Обучать отдельно на `train_audio`, затем дообучать на размеченных soundscape-сегментах.
- Добавить test-time augmentation: несколько crop/augment вариантов одного сегмента.
- Использовать site/time metadata из имен файлов как дополнительный prior.
- Делать class-wise calibration: часть видов не встречается в тесте, а часть встречается только в soundscape-разметке.

## Локальный запуск

Если датасет лежит не в стандартном Kaggle-пути, укажи переменную:

```bash
python3 -m pip install -r requirements.txt
export BC2026_DATA_DIR=/path/to/dataset
python notebooks/bc2026_baseline.py
```

Для быстрой проверки можно включить короткий режим:

```bash
export BC2026_FAST_DEV=1
python notebooks/bc2026_baseline.py
```

Для твоей локальной папки путь такой:

```bash
export BC2026_DATA_DIR="/Users/kseniadragun/Desktop/Predictive_Analytics/birds /birdclef-2026"
```

Быстрый обзор датасета:

```bash
export BC2026_DATA_DIR=/path/to/dataset
python src/eda_summary.py
```
