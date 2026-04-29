# BirdCLEF 2026 Pantanal Bioacoustics

Локальный ML-проект для Kaggle-соревнования по распознаванию видов животных в 5-секундных сегментах аудио из Pantanal.

## Что хранится в GitHub

- код обучения и инференса;
- EDA-скрипты;
- документация;
- `requirements.txt`.

Что не храним в GitHub:

- папки `train_audio/`, `train_soundscapes/`, `test_soundscapes/`;
- `.ogg` аудио;
- `submission.csv`;
- веса моделей (`.pt`, `.pth`, `.ckpt`).

Датасет весит много, поэтому каждый участник команды скачивает его локально с Kaggle и указывает свой путь через `BC2026_DATA_DIR`.

## Быстрый старт в VS Code

Открой эту папку в VS Code:

```bash
cd "/Users/kseniadragun/Documents/Codex/2026-04-28/overview-the-goal-of-this-competition"
```

Установи зависимости:

```bash
python3 -m pip install -r requirements.txt
```

Укажи путь к датасету:

```bash
export BC2026_DATA_DIR="/Users/kseniadragun/Desktop/Predictive_Analytics/birds /birdclef-2026"
```

Проверь, что данные читаются:

```bash
python3 src/eda_summary.py
```

Запусти быстрый тест baseline:

```bash
export BC2026_FAST_DEV=1
python3 notebooks/bc2026_baseline.py
```

В VS Code можно запускать без ручного ввода команд:

1. `Terminal` -> `Run Task...`
2. выбрать `BirdCLEF: EDA summary` или `BirdCLEF: Fast baseline`

Для первого локального запуска лучше выбирать `BirdCLEF: Fast baseline`.

Полный локальный запуск:

```bash
unset BC2026_FAST_DEV
python3 notebooks/bc2026_baseline.py
```

На MacBook Air полный запуск может быть медленным, потому что используется CPU. Для финального scoring всё равно понадобится Kaggle Notebook: скрытые `test_soundscapes/*.ogg` доступны только там.

## Структура

```text
.
├── docs/
│   ├── experiments.md
│   ├── github_workflow_ru.md
│   └── iteration_plan_ru.md
├── notebooks/
│   ├── bc2026_baseline.py
│   ├── bc2026_eda.ipynb
│   └── bc2026_experiments.ipynb
├── src/
│   └── eda_summary.py
├── README.md
├── README_RU.md
├── requirements.txt
└── .gitignore
```

## Текущий baseline

`notebooks/bc2026_baseline.py`:

- читает `train.csv`;
- добавляет размеченные сегменты из `train_soundscapes_labels.csv`;
- строит log-mel спектрограммы;
- обучает небольшую CNN;
- создает `submission.csv`.

Это стартовая версия, не финальная модель. Следующие улучшения описаны в `docs/iteration_plan_ru.md`.

Для работы по ячейкам в VS Code используй `notebooks/bc2026_experiments.ipynb`.

Для анализа данных и графиков используй `notebooks/bc2026_eda.ipynb`.
