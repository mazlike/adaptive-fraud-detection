# adaptive-fraud-detection

Прототип адаптивного конвейера ML для обнаружения мошенничества (fraud detection)
с оценкой влияния стратегий активного обучения (active learning) на качество модели
при сдвиге концепции (concept drift) и сильном дисбалансе классов с ограниченным бюджетом разметки.

Язык объяснений: русский. Код, комментарии, docstrings, идентификаторы, ключи конфига: английский.

## Гипотезы к проверке

| Гипотеза | Утверждение |
|---|---|
| **H1** | Без адаптации PR-AUC заметно падает после дрейфа; резкий дрейф hurts больше, чем постепенный. |
| **H2** | Дрейф-триггерное переобучение + активное обучение восстанавливает качество быстрее и с меньшим числом меток, чем базлайн без адаптации. |
| **H3** | При равном бюджете uncertainty sampling лучше случайного, но прирост ограничен при сильном дисбалансе (много неоднозначных легитимных транзакций, мало фрода). |
| **H4** | Rare-class-aware стратегия превосходит чистый uncertainty sampling по recall и PR-AUC, особенно при малом бюджете. |
| **H5** | Преимущество AL над случайным выбором сжимается по мере роста бюджета. |

Числа не выдумываются — гипотеза считается подтверждённой только после запуска экспериментов (этапы 1–6).

## Стек

Python 3.11+, numpy, pandas, scikit-learn, LightGBM, river (ADWIN), PyYAML, kagglehub, matplotlib, pytest.
Одна точка входа: `python -m afd.run --config configs/experiment.yaml`.

## Структура репозитория

```
adaptive-fraud-detection/
├── README.md                      # этот файл (инструкции по использованию)
├── pyproject.toml                 # метаданные проекта + зависимости
├── requirements.txt               # зафиксированные версии пакетов
├── configs/
│   └── experiment.yaml            # единый YAML-конфиг эксперимента
├── src/afd/
│   ├── __init__.py
│   ├── run.py                     # CLI: python -m afd.run --config configs/experiment.yaml
│   ├── data.py                    # загрузка IEEE-CIS (kagglehub) + синтетический генератор дрейфа
│   ├── metrics.py                 # PR-AUC, rolling PR-AUC, F1, recall@budget, recovery_time, detection_delay, false_alarms
│   ├── simulator.py               # преквенциальный симулятор (test-then-train), бюджет разметки
│   ├── baseline.py                # модель без адаптации + class weights / coût-чувствительное обучение
│   ├── detector.py                # ADWIN (river) + дрейф-триггерное переобучение (стадия 3)
│   ├── active_learning.py         # budget + стратегии random / uncertainty / rare_aware (стадия 4)
│   └── pipeline.py                # сборка конвейера из YAML (стадия 5)
├── tests/
│   ├── __init__.py
│   ├── test_config_loads.py       # стадия 0: конфиг + точка входа
│   ├── test_data.py               # стадия 1: генератор, дрейф в t0, без shuffling
│   └── test_simulator.py          # стадия 2: порядок test-then-train, бюджет не превышается, нет утечек
├── scripts/
│   ├── plot_results.py             # графики batch-level PR-AUC и стоимости разметки
│   └── analyze_sweep.py            # mean±std, ranking и paired Wilcoxon
└── docs/
    └── stage0_literature_and_metrics.md  # обзор методов, выбор метрик, допущения
```

## Быстрый старт

```powershell
# 1. Активировать виртуальное окружение
.\.venv\Scripts\Activate.ps1

# 2. Установить зависимости (версии зафиксированы в requirements.txt)
pip install -e .

# 3. Прогон тестов (должны все проходить)
pytest -q

# 4. Пробный запуск точки входа (стадия 0 — только загрузка и валидация конфига)
python -m afd.run --config configs/experiment.yaml
```

Ожидаемый вывод этапа 0:
```
[afd.run] config loaded: configs/experiment.yaml
[afd.run] experiment: stage0_smoke_test
[afd.run] Stage 0 smoke test: config is valid. Pipeline not yet implemented.
```

## Источник данных

По умолчанию используется **синтетический генератор** (`dataset.source: synthetic`), который не требует интернета и_accountów Kaggle. Он порождает хронологический поток с двумя классико-условными концептами, внедряет sudden или gradual drift в точку `t0` (сохраняется в `drift_points`) и калибрует долю фрода (0.1–2%).

Для реальных данных IEEE-CIS Fraud Detection укажите в конфиге:
```yaml
dataset:
  source: "ieee-cis"
  path: "C:\\путь\\к\\train_transaction.csv"   # локальный CSV, или оставьте null для kagglehub
  time_column: "TransactionDT"
  target_column: "isFraud"
```
Если `path` не задан, код вызовет `kagglehub.competition_download('ieee-fraud-detection')` — для этого нужны учётные данные Kaggle (`KAGGLE_USERNAME`/`KAGGLE_KEY` или `~/.kaggle/kaggle.json`). Строки сортируются по времени без перемешивания; колонки-идентификаторы удаляются.

## Конфигурация эксперимента

Единственный конфиг: `configs/experiment.yaml`. Ключи на английском, комментарии объясняют назначение и допустимые значения. Основные секции:

- `experiment` — имя, сид, output_dir
- `dataset` — источник (synthetic / ieee-cis), параметры генератора, тип дрейфа (`none` / `sudden` / `gradual`), `t0`, `width`, `interpolation` (`linear` / `sigmoid`), сид
- `simulator` — `batch_size`, `warmup_batches`
- `model` — `lightgbm` / `logistic`, `class_weight` (`none` / `balanced` / `manual`), `scale_pos_weight`, `cost_fp` / `cost_fn`, гиперпараметры
- `detector` — `enabled`, `type` (`adwin`), `delta`, `clock`, `input_signal` (`error` / `score`), `retrain_window`, `cooldown_batches`, `warmup_batches`
- `active_learning` — `enabled`, `budget_fraction` (0.01 / 0.05 / 0.10), `total_budget`, `strategy` (`random` / `uncertainty` / `rare_aware`), `uncertainty_measure`, параметры `rare_aware`
- `evaluation` — метрики, окно rolling PR-AUC, порог F1, уровень восстановления
- `seeds` — список сидов для многократных запусков
- `dataset.seed` — базовый seed synthetic stream; фактический seed потока равен
  `dataset.seed + experiment seed`, поэтому варианты при одном seed используют
  одинаковый stream, а разные seeds дают независимые paired streams.

## Прогон по стадиям

Репозиторий разработан поэтапно (одна стадия за раз). Состояние по стадиям:

| Стадия | Содержание | Статус | Как проверить |
|---|---|---|---|
| **0** | Обзор методов, метрики, структура репозитория, конфиг, точка входа | ✅ done | `pytest -q`, `python -m afd.run --config configs/experiment.yaml` |
| **1** | Загрузка данных + синтетический генератор дрейфа + тесты | ✅ done | `pytest tests/test_data.py -q` |
| **2** | Stream simulator, prequential test-then-train, baseline, баланс классов + тесты | ✅ done | `pytest tests/test_simulator.py -q` |
| **3** | ADWIN детектор + дрейф-триггерное переобучение | ✅ done | `pytest tests/test_detector.py -q` |
| **4** | Цикл активного обучения, 3 стратегии, бюджет | ✅ done | `pytest tests/test_active_learning.py -q` |
| **5** | Конфиг-конвейер, CLI, раннер экспериментов, скрипты графиков | ✅ done | `python -m afd.run --config configs/experiment.yaml` (полный прогон) |
| **6** | Таблицы, фигуры, выводы по H1–H5, ограничения, будущая работа | ✅ ready | `python scripts/analyze_sweep.py --results results/stage6_sweep` |

## Модульное тестирование

```powershell
.\.venv\Scripts\Activate.ps1
pytest -q                     # все тесты
pytest tests/test_data.py -q  # только стадия 1
pytest tests/test_simulator.py -q  # только стадия 2
pytest tests/test_detector.py -q  # только стадия 3
pytest tests/test_active_learning.py -q  # только стадия 4
```

Ключевые инварианты, которые покрыты тестами:
- Порядок test-then-train: после начального warm-up предсказание на батче происходит до обновления модели на этом батче.
- Warm-up используется только для инициализации модели и исключается из evaluation-метрик.
- Бюджет разметки никогда не превышается (ни доля в батче, ни общий лимит `total_budget`).
- Нет утечки будущего: модель обучается только на приобретённых метках из прошлого.
- Static baseline можно запустить с отключённым post-warm-up обновлением (`adaptation_enabled=False`).
- ADWIN получает `error` только для приобретённых меток, а `score` — для всех наблюдений; retraining использует только recent labeled window.
- Active learning поддерживает стратегии `random`, `uncertainty` (`margin` / `least_confidence` / `entropy`) и `rare_aware`; при отсутствии внешнего selector симулятор использует встроенный strategy-driven selector.
- Stage 5 сохраняет исходный конфиг, метрики по seed (`metrics_by_seed.csv/json`) и batch-level результаты (`batch_metrics.csv`) в `experiment.output_dir`; оптимальный режим выбирается только после фактического сравнения этих результатов.
- График результатов: `python scripts/plot_results.py --results results/stage5_pipeline`.
- Полный factorial sweep: `python -m afd.run --config configs/experiment.yaml --sweep`.
- Анализ sweep: `python scripts/analyze_sweep.py --results results/stage5_pipeline`.
- В baseline variant active learning и detector отключены; в `adwin` active learning отключён, а в `adwin_al` включены оба механизма. Это предотвращает смешение эффектов при проверке H1–H4.
- Все comparisons в paired Wilcoxon используют одинаковые seeds и один и тот же drift/budget/variant.
- Для `source: creditcard` необходимо явно указать локальный CSV path и
  соответствующие `time_column`/`target_column` (обычно `Time`/`Class`);
  автоматическая загрузка доступна только для IEEE-CIS через KaggleHub.
- Дрейф вставлен ровно в позицию `t0` (ground truth хранится в `drift_points`).
- Генерация детерминирована при фиксированном сиде.

## Пinned requirements

Версии зафиксированы после проверенной установки (`requirements.txt`):

```
numpy==1.26.4
pandas==2.3.3
scikit-learn==1.9.1
lightgbm==4.7.0
river==0.23.0
PyYAML==6.0.3
matplotlib==3.11.2
kagglehub==1.0.3
pytest==9.2.0
```

Обновить: `pip install -e .` → `pip freeze > requirements.txt` (в `.venv`).

## Допущения и ограничения

- Python 3.11+ (окружение проекта: 3.13; код совместим с 3.11 через `from __future__ import annotations`).
- Реальные данные через kagglehub требуют аутентификации Kaggle; при её отсутствии используется синтетический генератор.
- scikit-learn 1.9.1 / LightGBM 4.7.0 — версии из текущего индекса пакетов; при переносе окружения перепроверить совместимость.
- river 0.23.0: ADWIN создаётся через `ADWIN(delta=..., clock=...)`; smoke-тест импорта и API пройден.
- Глухие места в литературе помечены `[флаг]` в `docs/stage0_literature_and_metrics.md` — требуют верификации перед финальным отчётом.

## Лицензия

Внутренний прототип для исследовательских целей.
