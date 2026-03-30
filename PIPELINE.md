# SO-101 SmolVLA — полный пайплайн

Задача: **Put cube on plate** (pick-and-place) на роботе SO-101 в симуляторе MuJoCo.

---

## Структура проекта

```
LerobotHack-VLA/
├── collect_data/               # модуль сбора демонстраций
│   ├── config.py               # CollectDataConfig — все параметры сбора
│   ├── controller.py           # обёртка над мастер-рукой SO-101
│   ├── env_runner.py           # цикл записи эпизодов
│   └── run.py                  # точка входа
├── mujoco_env/
│   └── y_env.py                # SimpleEnv — основная среда симуляции
├── asset/
│   └── example_scene_y.xml     # MuJoCo сцена (робот + стол + объекты)
│
├── merge_demo_datasets.py      # слияние нескольких батчей в один датасет
├── analyze_demo_data_quality.py# метрики качества демо + badness_score  ⚠️ не протестирован
├── filter_demo_dataset_by_badness.py  # фильтрация по badness_score      ⚠️ не протестирован
├── trim_demo_dataset_start.py  # обрезка первых N кадров эпизода         ⚠️ не протестирован
│
├── run_official_smolvla_train_cached.sh  # обёртка docker run → lerobot.scripts.train
├── run_smolvla_inference.py    # инференс в симуляторе
├── run_checkpoint_benchmark.sh # прогон по всем чекпоинтам, таблица метрик
│
├── smolvla_compat.py           # загрузка SmolVLAConfig из train_config.json
├── smolvla_defaults.py         # пути по умолчанию (датасет, чекпоинт)
│
├── demo_data_merged_draft_hf/  # ФИНАЛЬНЫЙ датасет для обучения (262 эп., 113k кадров)
├── deprecated/                 # устаревший код (ноутбуки, старые скрипты)
└── outputs/
    ├── hf_cache/               # кэш HuggingFace (весов, токенайзера)
    └── train/
        └── so101_smolvla_official_main_bs32_lr1e4_noamp/
            └── checkpoints/    # 001000 … 017000, last
```

---

## 1. Сбор демонстраций

### Конфигурация (`collect_data/config.py`)

Все параметры сбора задаются в `CollectDataConfig`. Чтобы изменить их — редактируй значения по умолчанию прямо в датаклассе:

| Параметр | Значение по умолчанию | Описание |
|---|---|---|
| `repo_name` | `"so101_pnp"` | Имя датасета (LeRobot repo_id) |
| `num_demo` | `30` | Сколько эпизодов записать |
| `root` | `./demo_data4` | Папка, куда сохраняется датасет |
| `use_master_arm` | `True` | Управление мастер-рукой (`False` → клавиатура) |
| `leader_port` | `/dev/ttyACM0` | Порт мастер-руки |
| `motion_threshold` | `0.03` | Минимальное движение для старта записи |
| `task_name` | `"Put cube on plate"` | Текстовый промпт задачи |
| `xml_path` | `./asset/example_scene_y.xml` | MuJoCo сцена |
| `fps` | `20` | Частота записи |
| `image_size` | `(640, 480)` | Размер кадра в пикселях (W × H) |

### Запуск сбора

```bash
python -m collect_data
```

**Клавиши в окне симулятора:**

| Клавиша | Действие |
|---|---|
| `Z` | Сбросить сцену и отменить текущий эпизод |
| `X` | Принудительно сохранить текущий эпизод |
| *(автоматически)* | Эпизод сохраняется при детекции успеха (`check_success`) |

Запись начинается автоматически как только мастер-рука делает движение выше `motion_threshold`.

---

## 2. Подготовка датасета

### Актуальный датасет

> **Финальный датасет: `demo_data_merged_draft_hf/`**  
> 262 эпизода · 113 201 кадр · 20 fps  
> Признаки: `observation.image`, `observation.wrist_image`, `observation.state` (6-dim), `action` (6-dim)

Датасет используется **без какой-либо обработки и фильтрации** — собран, слит и сразу отправлен на обучение.

### Слияние батчей

Если набираешь данные в несколько сессий (`demo_data`, `demo_data2`, ...) — их нужно слить в один датасет:

```bash
python merge_demo_datasets.py \
    --inputs demo_data demo_data2 demo_data3 demo_data4 \
    --output demo_data_merged_draft
```

После слияния нужно конвертировать в HF-формат (LeRobot делает это при первом обращении к датасету через `LeRobotDataset`).

### Опциональная обработка (скрипты не протестированы)

> ⚠️ Скрипты ниже **не проверялись** на текущей версии датасета и пайплайна. Использовать с осторожностью.

**Анализ качества эпизодов** — считает `badness_score` и строит отчёт:

```bash
python analyze_demo_data_quality.py \
    --dataset-root demo_data_merged_draft \
    --output-dir outputs/analysis/
```

Результат: `outputs/analysis/episode_metrics.csv`, `report.md`, `report.html`, графики.

#### Как устроен `badness_score`

Оценка плохости — это взвешенная сумма нормализованных метрик, вычисленных по каждому эпизоду.

**Шаг 1. Сырые метрики** (из `observation.state`, `action`, `obj_init`):

| Метрика | Что измеряет |
|---|---|
| `length` | Число кадров в эпизоде |
| `idle_ratio` | Доля кадров, где EE двигался < 2 мм/шаг |
| `path_efficiency` | Прямое расстояние / длина пути (1.0 = прямая линия) |
| `turn_angle_mean_deg` | Средний угол между соседними векторами движения EE |
| `reversal_ratio` | Доля шагов, где EE развернулся назад (cos < 0) |
| `action_jerk_p95` | 95-й перцентиль рывка в командах (второй diff) |
| `gripper_toggles` | Число переключений gripper открыт↔закрыт |
| `regrasp_cycles` | Повторные захваты рядом с кубиком (`close_near_cube - 1`) |
| `far_close_events` | Сколько раз gripper закрылся далеко от кубика |
| `near_cube_entries` | Сколько раз EE заходил в зону кубика |
| `first_close_dist` | Расстояние до кубика в момент первого закрытия gripper |
| `first_close_progress` | Когда произошло первое закрытие (доля от длины эпизода) |

**Шаг 2. Нормализация — robust positive z-score**

Каждая метрика нормируется устойчиво (не чувствительно к выбросам):

```
scale = max(1.4826 × MAD,  IQR / 1.349,  std,  1.0)
z = clip((value − median) / scale,  0.0,  5.0)
```

Только положительные отклонения от медианы считаются плохими — z = 0 для лучших половины.

**Шаг 3. Взвешенная сумма:**

```
badness_score =
    1.3 × z_length               # слишком длинный эпизод
  + 1.1 × z_idle_ratio           # много простоя
  + 1.0 × z_toggles              # хаотичное управление gripper
  + 1.8 × z_regrasp_cycles       # повторные захваты (самый большой штраф)
  + 1.2 × z_far_close_events     # gripper закрывается мимо кубика
  + 1.0 × z_near_cube_entries    # много возвратов к кубику
  + 1.5 × z_first_close_dist     # первый захват промахивается
  + 1.0 × z_first_close_progress # первый захват очень поздно
  + 0.8 × z_action_jerk_p95     # дёрганые команды
  + 0.9 × z_turn_angle_mean_deg  # ломаная траектория
  + 0.8 × z_low_efficiency       # блуждающий путь
```

**Шаг 4. Рекомендации:**

| Условие | Вердикт |
|---|---|
| `badness ≥ p95` **или** ≥ 2 индивидуальных флага | `hard_drop_candidate` |
| `badness ≥ p90` (без hard_drop) | `review` |
| иначе | `keep` |

**Фильтрация по качеству** — удаляет эпизоды с `badness_score > max-badness`:

```bash
python filter_demo_dataset_by_badness.py \
    --dataset-root demo_data_merged_draft \
    --metrics-csv outputs/analysis/episode_metrics.csv \
    --max-badness 2 \
    --output-root demo_data_merged_badness_le2
```

**Обрезка начала эпизодов** — убирает первые N кадров (пауза перед движением):

```bash
python trim_demo_dataset_start.py \
    --dataset-root demo_data_merged_badness_le2 \
    --trim-frames 5 \
    --output-root demo_data_merged_badness_le2_trim5start
```

---

## 3. Обучение

### Предобученная база

Базовые веса скачиваются один раз и кэшируются в `outputs/hf_cache/`:

```bash
# Опционально — первый train-запуск сделает это сам
huggingface-cli download lerobot/smolvla_base \
    --local-dir outputs/models/lerobot_smolvla_base
```

Для обучения используется **sanitized** версия весов (без конфликтующих ключей):

```
outputs/models/lerobot_smolvla_base_sanitized_lr1e4/
```

### Запуск обучения (Docker)

```bash
./run_official_smolvla_train_cached.sh \
    --policy.path=/app/outputs/models/lerobot_smolvla_base_sanitized_lr1e4 \
    --dataset.repo_id=so101_pnp \
    --dataset.root=/app/demo_data_merged_draft_hf \
    --batch_size=32 \
    --steps=20000 \
    --output_dir=/app/outputs/train/so101_smolvla_official_main_bs32_lr1e4_noamp \
    --job_name=so101_smolvla_official_main_bs32_lr1e4_noamp \
    --policy.device=cuda \
    --policy.use_amp=false \
    --wandb.enable=false \
    --log_freq=50 \
    --save_freq=1000 \
    --num_workers=4
```

Скрипт — тонкая обёртка над `python -m lerobot.scripts.train` внутри контейнера `lerobot-workshop:latest`.  
Пути `/app/...` внутри контейнера = корень репозитория снаружи.  
HuggingFace-кэш монтируется из `outputs/hf_cache/` — веса **не скачиваются повторно**.

### Ключевые гиперпараметры

| Параметр | Значение |
|---|---|
| `batch_size` | 32 |
| `lr` | 1e-4 |
| `steps` | 20 000 |
| `use_amp` | `false` (без mixed precision) |
| `save_freq` | 1 000 шагов |
| `chunk_size` / `n_action_steps` | 50 |

### Артефакты обучения

```
outputs/train/so101_smolvla_official_main_bs32_lr1e4_noamp/
└── checkpoints/
    ├── 001000/pretrained_model/   # config.json, model.safetensors, train_config.json
    ├── 002000/pretrained_model/
    │   ...
    ├── 017000/pretrained_model/
    └── last -> 017000             # симлинк на последний
```

---

## 4. Инференс в симуляторе

### Одиночный запуск

```bash
xhost +local:docker   # разрешить Docker доступ к X11 (если нужен рендер)

docker run --rm --gpus all \
    --shm-size=16g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HOME=/root/.cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
    -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
    -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd):/app" \
    -v "$(pwd)/outputs/hf_cache:/root/.cache/huggingface" \
    -w /app \
    lerobot-workshop:latest \
    python run_smolvla_inference.py \
        --checkpoint-step 12000 \
        --episodes 5 \
        --max-steps 700 \
        --seed 42 \
        --device cuda \
        --summary-path /app/outputs/eval/run_ckpt12000.json
```

**Ключевые аргументы `run_smolvla_inference.py`:**

| Аргумент | Описание |
|---|---|
| `--checkpoint-step N` | загрузить чекпоинт `checkpoints/0N/pretrained_model` |
| `--train-run-dir PATH` | путь к папке обучающего прогона (по умолч. `outputs/train/so101_smolvla_official_main_bs32_lr1e4_noamp`) |
| `--policy-path PATH` | явный путь к `pretrained_model` (переопределяет выше) |
| `--episodes N` | количество эпизодов |
| `--max-steps N` | максимум шагов на эпизод |
| `--seed N` | seed для генерации сцен |
| `--headless` | запуск без окна (не нужен если X11 прокинут) |
| `--summary-path PATH` | куда сохранить JSON с метриками |

Скрипт автоматически:
- определяет `input_features` / `output_features` из `config.json` чекпоинта
- валидирует feature contract (ключи, типы, размерности)
- **не перекачивает** базовые VLM-веса (`load_vlm_weights=False` для локальных чекпоинтов)

### Бенчмарк по всем чекпоинтам

```bash
xhost +local:docker

CHECKPOINTS="1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000" \
EPISODES=20 \
MAX_STEPS=1000 \
./run_checkpoint_benchmark.sh
```

Результаты сохраняются в `outputs/eval/benchmark_20scenes_steps1000/`:
- `ckpt_NNNNNN.json` — результаты по каждому чекпоинту
- `summary.json` — сводная таблица
- `summary.md` — markdown-таблица `checkpoint | successes | success_rate | avg_steps`
