# Как работать командой через GitHub

## Один раз на компьютере

1. Открыть папку проекта в VS Code.
2. Установить зависимости:

```bash
python3 -m pip install -r requirements.txt
```

3. Скачать Kaggle-датасет локально.
4. В каждом новом терминале указывать путь:

```bash
export BC2026_DATA_DIR="/path/to/birdclef-2026"
```

У каждого участника путь может быть разный. Это нормально.

## Что коммитить

Коммитим:

- `.py` файлы;
- `README.md`;
- документы в `docs/`;
- `requirements.txt`;
- небольшие конфиги.

Не коммитим:

- аудио;
- скачанный датасет;
- `submission.csv`;
- большие веса моделей;
- временные файлы.

## Первый push в GitHub

Если репозиторий ещё не создан:

```bash
git init
git add .
git commit -m "Initial BirdCLEF baseline"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

Если репозиторий уже создан на GitHub, используй URL своей команды вместо `https://github.com/<USER>/<REPO>.git`.

## Обычная работа

Перед началом:

```bash
git pull
```

После изменений:

```bash
git status
git add .
git commit -m "Describe the change"
git push
```

## Рекомендуемое разделение задач

- Один человек улучшает модель в `notebooks/bc2026_baseline.py`.
- Один человек делает EDA и графики в `src/` или отдельном notebook.
- Один человек готовит Kaggle submission notebook.
- Один человек ведет README и фиксирует результаты экспериментов.

## Как сравнивать эксперименты

Заведите простую таблицу в README, Google Sheets или отдельном `docs/experiments.md`:

```text
date | author | model | data | local val loss | kaggle score | notes
```

Главное: каждый experiment должен быть воспроизводимым, то есть команда должна понимать, каким кодом и параметрами он был получен.
