# g-SpMM: turbo_gnn против DGL

Tesla T4, d=64, медиана из 7 прогонов по 30 запусков, CUDA events на обеих сторонах.
`ogbn-arxiv` (N=169 343, E=1 335 586) и `random` (N=200 000, E=3 400 000).
turbo — через `turbo_gnn.gspmm`, DGL — через `dgl.ops.*`, обе стороны на побитово
одинаковых данных. Прогон и методика: [gspmm_vs_dgl/](gspmm_vs_dgl/).

В скобках — сколько клеток из 18×графов оказались быстрее у DGL.

## fp32, без пайплайнинга

| kind | geomean | медиана | мин | проигр. | картинка | данные |
|---|---|---|---|---|---|---|
| forward | **1.89x** | 1.92x | 1.10x | 0/36 | [fp32_forward.png](fp32_forward.png) | [arxiv](gspmm_vs_dgl/results/results_ogbn-arxiv_float32_bwd.json) · [random](gspmm_vs_dgl/results/results_random_float32_bwd.json) |
| **backward** | **1.31x** | 1.18x | 0.44x | 5/36 | [fp32_backward.png](fp32_backward.png) | те же |
| fwd+bwd | 1.52x | 1.48x | 0.64x | 2/36 | [fp32_fwd_bwd.png](fp32_fwd_bwd.png) | те же |

## fp16, без пайплайнинга

| kind | geomean | медиана | мин | проигр. | картинка | данные |
|---|---|---|---|---|---|---|
| forward | **2.00x** | 2.07x | 1.12x | 0/36 | [fp16_forward.png](fp16_forward.png) | [arxiv](gspmm_vs_dgl/results/results_ogbn-arxiv_float16_bwd.json) · [random](gspmm_vs_dgl/results/results_random_float16_bwd.json) |
| **backward** | **0.68x** | 0.68x | 0.32x | 30/36 | [fp16_backward.png](fp16_backward.png) | те же |
| fwd+bwd | 0.99x | 0.98x | 0.48x | 21/36 | [fp16_fwd_bwd.png](fp16_fwd_bwd.png) | те же |

## fp32, pipeline_stages=2

Три графа вместо двух — `skewed` попал бонусом от прогона, который ушёл не на те
графы. Со stages=0 напрямую не сравнивается.

| kind | geomean | медиана | мин | проигр. | картинка | данные |
|---|---|---|---|---|---|---|
| forward | 1.97x | 2.09x | 0.71x | 5/54 | [fp32_stages2_forward.png](fp32_stages2_forward.png) | [arxiv](gspmm_vs_dgl/results/results_ogbn-arxiv_float32_bwd_st2.json) · [random](gspmm_vs_dgl/results/results_random_float32_bwd_st2.json) · [skewed](gspmm_vs_dgl/results/results_skewed_float32_bwd_st2.json) |
| backward | 1.18x | 1.14x | 0.24x | 14/54 | [fp32_stages2_backward.png](fp32_stages2_backward.png) | те же |
| fwd+bwd | 1.53x | 1.55x | 0.54x | 8/54 | [fp32_stages2_fwd_bwd.png](fp32_stages2_fwd_bwd.png) | те же |

## fp16, pipeline_stages=2

| kind | geomean | медиана | мин | проигр. | картинка | данные |
|---|---|---|---|---|---|---|
| forward | **1.57x** | 1.72x | 0.67x | 5/36 | [fp16_stages2_forward.png](fp16_stages2_forward.png) | [arxiv](gspmm_vs_dgl/results/results_ogbn-arxiv_float16_bwd_st2.json) · [random](gspmm_vs_dgl/results/results_random_float16_bwd_st2.json) |
| backward | 0.65x | 0.67x | 0.31x | 32/36 | [fp16_stages2_backward.png](fp16_stages2_backward.png) | те же |
| fwd+bwd | 0.91x | 0.93x | 0.45x | 24/36 | [fp16_stages2_fwd_bwd.png](fp16_stages2_fwd_bwd.png) | те же |

## Отдельный свип по пайплайнингу

fp32 forward, три графа, только turbo меняется. Данные:
[stages{0,1,2,4}.json](gspmm_vs_dgl/results/stages/) ·
[pipeline_stages.json](gspmm_vs_dgl/results/pipeline_stages.json).

| stages | geomean | проигр. | картинка |
|---|---|---|---|
| 0 | **2.15x** | 1/54 | [fp32_forward_stages0.png](fp32_forward_stages0.png) |
| 1 | 2.06x | 1/54 | [fp32_forward_stages1.png](fp32_forward_stages1.png) |
| 2 | 1.96x | 5/54 | [fp32_forward_stages2.png](fp32_forward_stages2.png) |
| 4 | 1.96x | 4/54 | [fp32_forward_stages4.png](fp32_forward_stages4.png) |

По 162 клеткам (три ширины) **ни одна** не стала быстрее с пайплайном:
stages=1 → 0.958x, stages=2 → 0.913x, stages=4 → 0.907x относительно stages=0,
худшая 0.669x. То же записано в докстринге `ReductionAggrKernel` по H100.

## fp32 forward на четырёх графах

[fp32_forward_4graphs.png](fp32_forward_4graphs.png) — самый первый прогон,
добавляет `skewed` и `ogbn-products` (126M рёбер, только `copy_u`): geomean 2.15x,
проигрывает только `copy_u/sum`. По 171 клетке всех ширин — geomean 2.14x.

---

# Выводы

**1. Forward стабильно 1.9–2.0x в обоих dtype, ни одного проигрыша.**

**2. Backward — узкое место, и в fp16 он разваливается: 0.68x против 1.31x в fp32.**
Каждая клетка в fp16 примерно вдвое хуже своей fp32-версии: `copy_e/sum`
0.80x → 0.41x, `add/sum` 0.91x → 0.55x, `div/min` 3.50x → 1.51x. Причина —
градиентные буферы всегда fp32, независимо от dtype операндов
([gspmm.h](../../csrc/spmm/gspmm.h)): при fp16-входах это вдвое больше байт на
atomicAdd плюс отдельный проход на каст в `_functions.py`, которого у DGL нет —
он копит в dtype операнда. Решение зашито осознанно («atomicAdd is neither
universally available nor accurate enough on fp16»), но цена оказалась выше
ожидаемой, а на sm_70+ выбор есть.

Этот разрыв виден только потому, что backward замеряется отдельно: в `fb` fp16
даёт обманчивые 0.99x — это forward 2.00x пополам с backward 0.68x.

**3. `mul/sum` — худшая клетка в обоих dtype** (0.44x fp32, 0.32x fp16), причина
другая: в [_functions.py](../../turbo_gnn/_functions.py) перед транспонированным
g-SpMM материализуется полная переставленная копия рёберных данных
(`rhs_t[bwd_edge_map]`) — gather на `E × d` при каждом вызове backward. Сюда же
`copy_e/sum` (0.80x) и `add/sum` (0.91x).

**4. Пайплайнинг вредит.** Сравнимая пара (fp16, два графа): forward
2.00x → 1.57x, backward 0.68x → 0.65x. Согласуется с отдельным свипом и с уже
задокументированным регрессом на H100.

## Что чинить, по приоритету

1. **Тип аккумулятора по dtype операнда** вместо жёсткого fp32 — вернёт
   fp16-backward с 0.68x в район 1.3x. Если fp32-точность нужна осознанно —
   хотя бы убрать каст-проход, записывая финальный dtype прямо из ядра.
2. **Косвенное чтение `rhs` через `bwd_edge_map` внутри транспонированного ядра**
   вместо материализации копии — вылечит `mul/sum` и `div/sum`.
3. **Не включать пайплайнинг**: измеренно хуже на двух GPU, двух семействах ядер
   и обоих dtype.

## Чего в этих числах нет

Только d=64 на картинках (в JSON есть 32 и 128). Только `ogbn-arxiv` и `random`,
кроме отдельно помеченных строк. Фиксированные `warps=8, fpb=32, tiles_y=8`, без
автотюна. Сверка с DGL проходила на всех клетках при stages=0; для stages=2 она
выключена (`CHECK=0`) — те же данные проверены при stages=0, но формально это
непроверенный прогон. `ogbn-products` только в четырёхграфовом forward: при 126M
рёбер операнд `[E, d]` не влезает в 15 GB видеопамяти.
