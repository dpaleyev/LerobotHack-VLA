# Deprecated

Файлы, которые больше не используются в основном пайплайне.

| Файл / папка | Причина |
|---|---|
| `train_smolvla.py` | Кастомный тренировочный скрипт; заменён официальным `lerobot.scripts.train` через `run_official_smolvla_train_cached.sh` |
| `SMTH.md` | Черновые заметки; заменён `PIPELINE.md` |
| `y_env2.py` | Старая версия среды симуляции; актуальная — `mujoco_env/y_env.py` |
| `visualize_data/` | Модуль визуализации данных; заменён ноутбуком `2.visualize_data_ru_so101.ipynb` |
| `1.collect_data_ru_master.ipynb` | Jupyter-вариант сбора данных; заменён модулем `collect_data/` |
| `2.visualize_data_ru_so101.ipynb` | Jupyter-визуализация; заменён модулем `visualize_data/` (тоже здесь) |
