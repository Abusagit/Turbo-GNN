# Load-imbalance simulator: from an empty machine to the figures

The simulator charges a vertex `alpha + beta * degree` ticks and schedules the resulting
blocks onto SM slots. `beta` is fixed at 1 by definition -- one tick is the cost of one
neighbour -- so the whole model rests on two numbers measured from the real kernels: `alpha`,
the per-vertex prologue and epilogue, and the duration of a tick.

Both belong to the card they were measured on. **Measure and simulate on the same GPU**, or
the run silently mixes two machines.

## The pipeline

| # | step | script | needs a GPU | cost |
|---|---|---|---|---|
| 1 | time the real kernels | `measure_kernel_times.py` | **yes** | ~1 min |
| 2 | fit the cost model | `calibrate_cost_model.py` | no | seconds |
| 3 | sweep the grid | `sweep_all.py` | no | hours |
| 4 | draw | `plot_heatmap_grid.py` and friends | no | minutes to hours |

Only step 1 touches the GPU. Everything after it is CPU-bound and can run while the card is
busy with something else.

### 1. Measure

```bash
python scripts/ablation/measure_kernel_times.py \
    --graphs cora citeseer pubmed tolokers-2 ogbn-arxiv ... \
    --warmup 10 --iters 30 --out reports/measurements.jsonl
```

Times `gt`, `gat_v2` and `min_aggr`, forward and backward, at head dimensions 128 and 256,
on every graph given, using CUDA events. One JSON record per (graph, conv, pass, head dim)
with the graph's node and edge counts, the milliseconds per launch and the device name. A
graph that will not load is skipped with its reason and the run continues.

### 2. Calibrate

```bash
python scripts/ablation/calibrate_cost_model.py \
    --from-json reports/measurements.jsonl --out cost_models.json
```

Regresses `t(N, E) = c + a*N + b*E` per (conv, pass, head dim), under `a, b >= 0`. Since
`sum(alpha + beta * degree) = alpha*N + beta*E`, normalising `beta` to 1 gives
`alpha = a/b` and a tick of `b` nanoseconds. Prints a table: `alpha`, ns per tick, launch
overhead, R2, median error, and what share of the time edges explain.

Read the "do not predict wall clock from these cells" list. A cell is flagged when `alpha`
hits the non-negativity constraint, when edges explain under 20% of its time, or when the
median error exceeds 35%. Flagged cells are skipped by every later step. One or two flags is
normal -- min-aggregation's backward pass is O(N), not O(E), so no `alpha + beta * degree`
model describes it.

`cost_models.json` is **not** committed: it is only valid for the card it was measured on.

### 3. Sweep

```bash
python scripts/ablation/sweep_all.py --cost-model cost_models.json \
    --sms 108 --memory-bandwidth-gbps 2039 --out reports/sweep.csv
```

One row per (graph, cell, quantile, launch mode, split-K): makespan `T`, the perfect-packing
time `T*`, which of the three lower bounds attains it, the ratio, the drain tail, mean slot
occupancy and predicted milliseconds. This is where the conclusions come from; the figures
only illustrate them.

Resumable -- rows already in the CSV are skipped -- and dominated by the graphs with the
largest hub, since a run without slicing lasts about as long as the biggest vertex.

### 4. Draw

```bash
# Five panels per figure, as PDF, for the paper.
python scripts/ablation/plot_heatmap_grid.py \
    --setups tolokers-2:gat_v2:forward:128:0.99 ogbn-arxiv:gt:forward:128:0.99 \
    --cost-model cost_models.json --sms 108 --memory-bandwidth-gbps 2039 \
    --out-dir reports/figures

# Two panels: the bucketed baseline against concurrent + edge slicing.
python scripts/ablation/plot_heatmap_pair.py --dataset tolokers-2 \
    --cost-model cost_models.json --conv gat_v2 --pass forward --head-dim 128 \
    --sms 108 --memory-bandwidth-gbps 2039 --out pair.png
```

`plot_heatmap_grid.py` prints the graph's shape and every panel's `T`, `T*` and `T/T*`, which
is everything a caption needs. Pass `--tex` if you also want that written as LaTeX.

The older figure scripts are still here: `plot_all.py` walks the whole grid into per-cell
directories of PNGs, `plot_launch_modes.py` draws one such figure, and
`plot_occupancy_profile.py` collapses the SM axis into a single occupancy curve.

## Exploring a single configuration

`simulate_load_imbalance.py` is the layer underneath all of the above, and runs one point of
the space with every knob exposed -- assignment order, vertices per block, blocks per SM,
occupancy, launch mode, slice size. Start here when changing the model rather than using it.
`dump_timeline.py` prints the resulting schedule block by block.

## Checking that the simulation is sound

Three things are worth re-deriving after any change to the cost model.

**The tick should track memory bandwidth.** It is a per-neighbour cost, and that cost is
memory-bound, so moving to a card with `k` times less bandwidth should multiply it by `k`.
Between an H100 and an A100 the measured ratio was 1.62 against a bandwidth ratio of 1.64.
Nothing in the calibration knows about bandwidth, so this is an independent check.

**`alpha` should be roughly card-independent.** It is a ratio of two fitted coefficients, so
a uniform slowdown cancels. Large moves in `alpha` between cards mean the fit is unstable,
not that the hardware changed.

**`T/T*` should never fall below 1.** It is a makespan over a lower bound on the same work;
a value under 1 means a bound is computed wrong.

Unit tests cover the fit and the launch barrier:

```bash
python -m pytest tests/unit/test_cost_model_calibration.py \
                 tests/unit/test_load_imbalance_simulation.py
```

## Runbooks

`dev/sinfillo/recalibrate.sh` runs steps 1--4 end to end on a new card.
`dev/sinfillo/paper_figures.sh` runs step 4 for the setups chosen for the paper.

## DRDo: несколько бакетов без degree ordering

Эксперимент реализован в `sweep_degree_buckets.py` и `turbo_gnn/bucket_simulation.py`.
Это **симуляция расписания**, калибруемая замерами настоящих CUDA-ядер. Новые CUDA-ядра
с произвольным числом бакетов здесь не реализованы, поэтому `predicted_ms` нельзя выдавать
за измеренное ускорение GPU.

### Как работает текущая симуляция

1. `load_real_degrees` загружает граф через `src.data.datasets`, с принятым в проекте
   добавлением self-loops, строит CSR и извлекает степени. Forward и backward используют
   разные направления CSR.
2. Время блока — `round(alpha + beta * degree)`, минимум один tick. Калибровка на нескольких
   графах подбирает `t = c + a*N + b*E`, затем `alpha=a/b`, `beta=1`.
3. Блоки занимают долю SM, определяемую `max_blocks_per_sm * occupancy`. Планировщик случайно
   выбирает доступный SM, а ограничитель памяти разрешает продвижение ограниченного числа
   блоков за tick. Это упрощённая модель; регистры, кэш и реальные CUDA scheduler policies
   напрямую не воспроизводятся.
4. Исходный эксперимент использовал light/heavy, последовательный или concurrent запуск,
   а также опциональные slicing и merge. Внутри бакетов порядок уже был естественным.
   `assignment=lpt` перераспределяет вершины между блоками и не равнозначен простому
   degree ordering блоков по убыванию степени.
5. В новом эксперименте один блок обрабатывает одну вершину/голову, slicing выключен.
   Общий tick привязан к fitted work time двухбакетного sequential/natural baseline.
   Фитированный intercept `c` добавляется один раз к каждому варианту. Это прогноз
   регрессии, а не привязка к фактически измеренному времени конкретного графа.

Исправлен `sequential` в `turbo_gnn/simulation.py`: он теперь ждёт завершения всех
предыдущих запусков, в том числе с именами `bucket_0`, `bucket_1`, … . Старый код
обеспечивал эту границу только для специальных имён `heavy/light/merge`.

### Варианты на выбор

| Вариант | Что делим | Для чего |
|---|---|---|
| `quantile`, 2/4/8 бакетов | Light на примерно равные по числу вершин группы; heavy остаётся целым | Начальный контроль: меняется только число диапазонов |
| `log`, 2/4/8 бакетов | Light по логарифмическим диапазонам степени; heavy остаётся целым | Более узкие диапазоны маленьких степеней |
| `tail`, 2/4/8 бакетов | И light, и heavy, с более мелкими группами в хвосте распределения | Проверка замены полного degree ordering грубыми диапазонами |

Для `tail`, 4 бакетов и `q=.99` границы равны квантилям `.9/.99/.999`.
Число бакетов для `tail` должно быть чётным: тогда исходный heavy-порог сохраняется.
У каждой вершины остаётся исходный класс ресурсов: по умолчанию 8 light или 4 heavy
блока на SM. Это параметры модели, а не измеренная CUDA occupancy.
Равные степени не разрываются между бакетами; пустые бакеты не запускаются.
Поэтому в CSV есть и `requested_buckets`, и `effective_buckets`.

По умолчанию `--orderings natural`: **degree ordering выключен внутри всех экспериментальных
бакетов**; исходные node IDs сохраняются. Сами диапазоны запускаются от больших степеней
к маленьким. Двухбакетный `descending` автоматически добавляется как контроль. Для полного
факторного сравнения можно указать `--orderings natural descending`.

`sequential` включает барьер между бакетами; `concurrent` делает все бакеты доступными
одновременно. `--extra-launch-us 0 2 5` добавляет к прогнозу 0/2/5 мкс за каждый непустой
бакет сверх baseline. Это **анализ чувствительности**, не измеренная цена запуска;
для concurrent это упрощённая сериализованная надбавка. Время подготовки бакетов,
сортировки, locality и настройки ядер по бакетам не моделируется.

### Датасеты из основной части статьи

`10897_Faster_Generalized_Messa.pdf`, таблицы 1–3, страницы 7–8: **16 графов**.
Готовый список доступен как `--graphs paper`:

```text
artnet-exp avazu-ctr citeseer city-reviews city-roads-L city-roads-M
cora hm-categories ogbn-arxiv ogbn-products ogbn-proteins pokec-regions
pubmed tolokers-2 twitch-views web-fraud
```

Для первого расширения после smoke: `tolokers-2 ogbn-arxiv web-fraud city-roads-L pubmed`.
Это сочетает графы, на которых статья отмечает пользу ordering/bucketing, с дорожным
контролем и небольшим графом. Все 16 не нужны для первого теста: `ogbn-products` и
`ogbn-proteins` крупные и в локальном кэше этого запуска отсутствовали.

Дополнительно можно проверить `web-traffic` (уже есть локально), `amazon-computers`,
`coauthor-physics`, `reddit` (поддерживаются загрузчиком; при отсутствии кэша нужен download),
а также `synth-N16384` / `synth-N65536` как контролируемый stress-test. Синтетические графы
стоит использовать для проверки расписания в ticks, а не для фитирования cost model:
их locality отличается от реальных графов. Прогноз миллисекунд на них — экстраполяция.

### Команды для A100

Из корня репозитория, в окружении проекта с рабочим CUDA extension и зависимостями
загрузчика. По умолчанию используется `.venv/bin/python`; для другого окружения задайте `PY`.
Сбор занимает восемь реальных графов и охватывает `gt/gat_v2/min_aggr`, forward,
одну голову, d=128/256, **fp32**, q=.99. Статья использует fp16, а Table 1 — 2 головы
по 64; эти команды воспроизводят протокол существующей калибровки симулятора,
а не численные результаты Table 1.

После перехода со старой ветки на `drdo-main` пересоберите CUDA extension именно
из этой рабочей копии: в актуальном main изменились ядра и их ABI. Команда ниже
предполагает установленное окружение проекта, CUDA toolkit и зависимости сборки.
В новом checkout сначала установите окружение по инструкции корневого README.

```bash
# 0. На A100 после переключения ветки: пересобрать extension текущего checkout.
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 TURBO_GNN_FORCE_BUILD=TRUE \
  .venv/bin/python -m pip install -e . --no-build-isolation

# 1. На машине с A100: измерить ядра и получить cost_models.json одной командой.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  bash scripts/ablation/calibrate_device.sh a100 reports/drdo-a100

# 2. На CPU: сравнить 2/4/8 бакетов, три стратегии, два режима запуска, три seed.
OMP_NUM_THREADS=4 .venv/bin/python scripts/ablation/sweep_degree_buckets.py \
  --hardware a100 --cost-model reports/drdo-a100/cost_models.json \
  --graphs tolokers-2 --conv gat_v2 --head-dim 128 \
  --orderings natural --plot --out reports/drdo-a100/sweep

# 3. Расширить на все датасеты из основной части статьи (значительно дольше).
OMP_NUM_THREADS=4 .venv/bin/python scripts/ablation/sweep_degree_buckets.py \
  --hardware a100 --cost-model reports/drdo-a100/cost_models.json \
  --graphs paper --conv gat_v2 --head-dim 128 \
  --orderings natural --out reports/drdo-a100/paper
```

Профиль `a100` означает A100 SXM 80GB: 108 SM, 2039 GB/s; `h100` — H100 SXM 80GB:
132 SM, 3350 GB/s. Для другой модификации явно передайте `--sms` и
`--memory-bandwidth-gbps`. Скрипт измерений проверяет тип реально видимой карты.
Калибратор сохраняет GPU/SM/dtype/heads/quantile и отказывается смешивать несовместимые
измерения. Новый sweep проверяет эти метаданные и отклоняет ненадёжные ячейки fit.
Старые JSON без метаданных для этого sweep нужно пересобрать, а не переименовать.
После обновления CUDA-ядер заново соберите cost models даже на той же карте.
Скрипт также проверяет, что extension загружен из текущего checkout: editable-установка
другой рабочей копии может незаметно подставить старый бинарник. При обновлении ядра
внутри одной папки эта проверка не заменяет пересборку.

Если нужны только данные и fit, достаточно шага 1. Файлы:
`measurements.jsonl`, `measurements.log`, `calibration.log`, `cost_models.json`.
Для более короткого сбора той же ячейки добавьте к команде шага 1
`--convs gat_v2 --head-dims 128`.
Для расширенной калибровки можно добавить `--graphs` со своим списком реальных графов.
Загрузчик может пропустить недоступный граф — смотрите лог и `num_samples` в модели.
Не используйте ячейки, перечисленные калибратором под `do not predict wall clock`.

Новый sweep намеренно работает с forward: существующий `measure_kernel_times.py --passes backward`
измеряет forward+backward вместе. Считать этот замер стоимостью только backward было бы неверно.

### Тестовый прогон H100, 2026-11-09

Это исторический прогон **до переноса на main**, на базе `54a6673`
(`cost-model-calibration`). Исходный DRDo сохранён в ветке `drdo-snapshot`;
его код перенесён на `origin/main` (`2e657af`) в `drdo-main`.
Приведённые ниже времена не являются измерениями новых ядер main.
Артефакты находятся в исходной рабочей папке и не включены в Git.

Реальные CUDA-замеры проведены на свободной H100 №1, NVIDIA H100 80GB HBM3, 132 SM,
PyTorch 2.11.0+cu129. Сохранены отдельно от ранее существовавших cost models:
`reports/drdo-h100-smoke/measurements.jsonl` и `cost_models.json`.
Восемь графов × GT/GATv2 forward × d128 = 16 измерений; warmup=10, iters=30, fp32.
У GATv2 fit медианная ошибка около 19%, у GT около 10%; для ogbn-arxiv ошибки
достигают 65%/72%, поэтому абсолютные миллисекунды — только ориентир.

На `tolokers-2`: N=11 758, E=1 049 758 после self-loops, max degree=2 139.
Реальное время исходного GATv2 forward — **0.3833 ms**. Его регрессионный baseline
в симуляторе — **0.2752 ms** (примерно 28% ниже замера). Следующая таблица —
**только прогноз симулятора**, среднее по seed 42/43/44, без надбавки за дополнительные launches:

| Вариант | Sequential, ms | Concurrent, ms |
|---|---:|---:|
| 2 бакета, ordering OFF | 0.2752 | 0.1701 |
| 2 бакета, ordering ON, контроль | 0.2295 | 0.1701 |
| 4 quantile, ordering OFF | 0.2767 | 0.1701 |
| 8 quantile, ordering OFF | 0.2827 | 0.1701 |
| 4 log, ordering OFF | 0.2784 | 0.1701 |
| 8 log, ordering OFF | 0.2994 | 0.1701 |
| 4 tail, ordering OFF | 0.3668 | 0.1701 |
| 8 tail, ordering OFF | 0.6031 | 0.1701 |

На этом графе **больше бакетов не ускоряет**: sequential получает дополнительные
барьеры, concurrent уже достигает critical-path bound = 2140 ticks одной вершины.
С надбавкой 2 мкс за запуск concurrent-прогноз ухудшается с 0.1701 до 0.1741/0.1821 ms
для 4/8 бакетов. Это результат одного графа в данной модели, а не вывод для всей выборки.

CSV и картинки: `reports/drdo-h100-smoke/sweep/` (quantile/log, включая ordering ON)
и `reports/drdo-h100-smoke/sweep-tail/` (tail, ordering OFF и двухбакетные контроли).
`speedup_vs_2_natural` и `speedup_vs_2_ordered` сравнивают одинаковый launch mode и seed;
значение больше 1 означает ускорение. `manifest.json` фиксирует аргументы, SHA256 модели
и допущения. Параметры 8/4 resident blocks и 400 ns memory latency заданы моделью;
в этом smoke bandwidth cap не ограничивает работу.

Запустить тот же протокол (на `drdo-main` сначала пересоберите extension с
`TORCH_CUDA_ARCH_LIST=9.0`; времена новых ядер могут отличаться от исторического smoke):

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
  bash scripts/ablation/calibrate_device.sh h100 reports/drdo-h100-repeat \
  --convs gt gat_v2 --head-dims 128

OMP_NUM_THREADS=4 .venv/bin/python scripts/ablation/sweep_degree_buckets.py \
  --hardware h100 --cost-model reports/drdo-h100-repeat/cost_models.json \
  --graphs tolokers-2 --strategies quantile log --orderings natural descending \
  --plot --out reports/drdo-h100-repeat/sweep

OMP_NUM_THREADS=4 .venv/bin/python scripts/ablation/sweep_degree_buckets.py \
  --hardware h100 --cost-model reports/drdo-h100-repeat/cost_models.json \
  --graphs tolokers-2 --strategies tail --orderings natural \
  --plot --out reports/drdo-h100-repeat/sweep-tail
```
