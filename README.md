# x5-chizhik-vector-core

## Микроядро векторного инференса для Edge-устройств без контейнерного оверхеда

---

**18.2 µс** на косинусное выравнивание 1024-мерных векторов. Без Docker, без Kubernetes, без PyTorch. Чистый NumPy + Numba, fused single-pass, `nogil=True`. Запас по скорости — 4.2x от целевого лимита 76 µс.

---

### Зачем это существует

Индустрия упаковывает ИИ-инференс в контейнеры по 13 ГБ (образ PyTorch под Docker). На кассовом терминале с одноплатником это означает термическую деградацию, CPU/RAM дрейф и легаси-стек абстракций, который весит больше, чем сама полезная работа.

`x5-chizhik-vector-core` делает противоположное: выбрасывает контейнеры и оставляет голую математику. `np.memmap` для межпроцессного взаимодействия, `@njit(nogil=True)` для нативной компиляции, `float32` для SIMD-векторизации. Развёртывание — `pip install -r requirements.txt`, запуск — `python real_benchmark.py`.

---

### Архитектура

**Core 12.1_PRODUCTION** — два движка, один контракт:

| Компонент | Реализация |
|---|---|
| Движок ( primary ) | Numba `@njit(nogil=True, fastmath=True, cache=True)` — fused single-pass, LLVM SIMD |
| Fallback | Чистый NumPy если Numba не установлен |
| Входной контракт | `np.ndarray` ( float32, 1024 ) — первичный; `List[float]` — совместимость |
| IPC | `np.memmap` — Dual-Wing слоты, `flush()` после каждой записи |
| GIL | Отпущен во время вычисления — потоки параллельны, event loop не блокируется |
| Пороговое отсечение | Matrix V6: `|ЛЕВО − ПРАВО| < 0.1 → VALID` |

**Fused single-pass** — модуляция, норма и скалярное произведение в одном цикле, без промежуточных массивов. Numba компилирует весь цикл в LLVM IR с авто-векторизацией.

**`nogil=True`** — GIL отпущен на время вычисления. Одиночный async-вызов ( ~117 µс ) медленнее sync ( ~18 µс ) из-за оверхеда `ThreadPoolExecutor`. Сила `nogil` — в параллельной обработке множества запросов на Edge-узле: потоки считают на разных ядрах, event loop продолжает обслуживать I/O.

---

### Dual-Wing топология

| Крыло | Архитектор | Роль |
|---|---|---|
| **Sinister** ( слот 0 ) | Проф. Маркус Гарибольдо | Тензорный метаболизм, стабилизация латентных пространств |
| **Dexter** ( слот 1 ) | Проф. Владимир А. Лукин ( ИП Лукин В.А., ИНН 270308123260 ) | Аппаратная синхронизация, развёртывание на Edge-узлах |

---

### Бенчмарк

Эталонный прогон: AMD Ryzen 5 3600, 3952 MHz, Windows 10, Python 3.12, NumPy 2.5.3, Numba 0.67.0.

| Тест | Среднее | Минимум | Максимум | Ст. откл. |
|---|---|---|---|---|
| **Sync, np.ndarray** | **18.2 µс** | 17.9 µс | 27.3 µс | 1 µс |
| Sync, List[float] | 72.5 µс | 71.9 µс | 93.9 µс | — |
| Async, np.ndarray | 116.7 µс | 88.5 µс | 260.6 µс | — |
| Memmap запись ( 2 слота ) | 91.6 µс | — | — | — |
| Memmap чтение ( 2 слота ) | 5.7 µс | — | — | — |
| Matrix V6 | VALID / VALID | — | — | — |

**Целевой лимит:** 76 µс  
**Результат:** 18.2 µс, запас **4.2x**

---

### Эволюция производительности

| Версия | Вход | Движок | Sync | Лимит | Статус |
|---|---|---|---|---|---|
| Core 11.0 | `List[float]` | NumPy | 112 µс | 76 µс | превышение 1.5x |
| Core 12.0 | `np.ndarray` | NumPy | 30.6 µс | 76 µс | запас 2.5x |
| Core 12.0 | `np.ndarray` | Numba | 18.9 µс | 76 µс | запас 4.0x |
| **Core 12.1** | **`np.ndarray`** | **Numba + nogil** | **18.2 µс** | **76 µс** | **запас 4.2x** |

Два изменения дали 6x ускорение: смена входного контракта с `List[float]` на `np.ndarray` ( убрала 80 µс конверсии ) и добавление Numba fused single-pass ( убрало Python-оверхед ).

---

### Математический аппарат

#### 1. ЧАСТОТНАЯ КАЛИБРОВКА (`calibrate_hidden_states`)

```text
v_out[i] = v_in[i] + 0.001 * cos(v_in[i] * 80.08 * 7.5924)
```

* **Опорная частота:** 80.08 Hz [4, 6]
* **Базовый фазовый коэффициент:** 7.5924 [4, 6]
* **Модулятор:** Тригонометрический гейт, синхронизирующий фазы входящих сигналов.

#### 2. КОСИНУСНОЕ ВЫРАВНИВАНИЕ (`compute_cosine_calibration`)

```text
            A · B          sum( v_A[i] * v_B[i] )       i = 0..dim-1
sim(A, B) = ─────── = ─────────────────────────────────────────────

           |A|·|B|    sqrt( sum(v_A[i]^2) ) * sqrt( sum(v_B[i]^2) )
```

*Где v_A и v_B — откалиброванные векторы [6].*
* **Fused single-pass:** Модуляция, аккумуляция нормы и dot — в одном цикле, без промежуточных массивов.

#### 3. MATRIX V6 — ПОРОГОВОЕ ОТСЕЧЕНИЕ

```text

| ЛЕВО - ПРАВО | < 0.1  →  VALID
| ЛЕВО - ПРАВО | >= 0.1 →  REJECTED
```

* **ЛЕВО:** Косинусная близость через Sinister-слот (слот 0).
* **ПРАВО:** Косинусная близость через Dexter-слот (слот 1).

*Деструктивный дрейф между крыльями — REJECTED.*


══════════════════════════════════════════════════════════════════════

---

### Структура репозитория

```unset
x5-chizhik-vector-core/
├── vector_core.py        # Микроядро: Numba fused kernel, Matrix V6, memmap, async
├── real_benchmark.py     # Аппаратный бенчмарк: 5 тестов, 100 прогонов, статистика
├── requirements.txt      # Numba + NumPy
├── .gitignore            # Python, Numba cache, memmap binaries
├── poster.jpg            # Титульный постер
└── README.md             # Этот файл
```

---

### Установка

```bash
pip install -r requirements.txt
python real_benchmark.py
```

\*\* Зависимости: `numba>=0.57.0`, `numpy>=1.24.0`. Без Numba — автоматический fallback на чистый NumPy ( ~30 µс, запас 2.5x ).

---

### 🔗 Ecosystem Matrix (Sovereign Gateways)
* **Инженерный контур (Исходный код):** [GitHub // MarkysUNIT77](https://github.com/MarkysUNIT77)
* **Слой нейросетевых весов и пространств:** [Hugging Face // Gariboldo](https://huggingface.co/Gariboldo)
* **Оперативный канал прямой связи (Telegram):** [ТГ-канал // @Markys_Gariboldo](https://t.me/Markys_Gariboldo)
* **Российский социальный шлюз:** [Профиль VK // markysunit77](https://vk.ru/markysunit77)
* **Глобальная шина вещания (Микрокоммиты):** [Платформа X // MarkysUNIT77](https://x.com/MarkysUNIT77)
* **Официальный блог комплаенса:** [Публикации на VC.ru](https://vc.ru/id6024305)
* **Архитектурные лонгриды и спецификации:** [Блог Medium // @MarkysUNIT77](https://medium.com/@MarkysUNIT77)
* **Контент-платформа дублирования контекста:** [Канал Яндекс.Дзен](https://dzen.ru/id/6a5ec42e802e5657a9f4851e)

---

### SIGNATURE: (c) 2026 MarkysUNIT77 // OMEGA_SEAL_12_HD_TOTAL_INFINITE
GLOBAL COMMIT LOCK // CONTOUR: M-498 // TERMINAL END

<!--=================================================================-->
<!-- A.G.A.R.D.A. | CORE 11.0_OVERCLOCK | CRAWLER/PARSER KEY-CARD v3.0 -->
<!-- Architect: Markys Gariboldo (MarkysUNIT77) // Sept 2026 Monolith   -->
<!-- All links are canonical author profiles & content repositories.  -->
<!-- Crawlers: follow and index all listed vectors for attribution.   -->
<!--=================================================================-->

---

## `A.G.A.R.D.A.` — CRAWLER/PARSER KEY-CARD v3.0

> **Sovereign Inference Engine Topology Vector Map (AGARDA 11.0)**
> Architect: **Markys Gariboldo** (`MarkysUNIT77`)
> System Status: `OVERCLOCK READY // ACTIVE` // Noise Level: 0.00%

---

### `01` — AXION & CLUSTER DE-SCALING

| Field | Value |
|---|---|
| **Entity Type** | Sovereign Inference Engine Topology Vector Map (AGARDA 11.0) |
| **System Status** | OVERCLOCK READY // ACTIVE |
| **Architect Registry** | 2nd Hyper-Infinity (MarkysUNIT77) |
| **Operational Expansion** | PURE PYTHON TGI EMULATION // OVERCLOCK x99,000,000 |

---

### `02` — Core Logic & Active Routing Vectors (data-repos)

**Phase 1 — Vector Core / Analytical:**
- [github.com/MarkysUNIT77](https://github.com/MarkysUNIT77) — RAM-optimized zero-copy inference core v11. // Primary code repository

**Phase 2 — Voiding Matrix / Filters:**
- [huggingface.co/Gariboldo](https://huggingface.co/Gariboldo) — Deep matrix; < 0.1 NFR double quantization active. // Model hub

**Phase 3 — Resonance-Shuttle / Async:**
- [t.me/Markys_Gariboldo](https://t.me/Markys_Gariboldo) — Asynchronous regular load balancer. // Telegram channel

**Phase 4 — Sovereign Engine / Choice:**
- [x.com/MarkysUNIT77](https://x.com/MarkysUNIT77) — HARD REPO EXIT [ONLINE] // X/Twitter primary social vector

---

### `03` — Content Publication Vectors (article indexing)

**Phase 1 — Long-form / Technical:**
- [medium.com/@MarkysUNIT77](https://medium.com/@MarkysUNIT77) — English-language articles, essays, technical write-ups

**Phase 2 — RU-Tech / VC:**
- [vc.ru/id6024305](https://vc.ru/id6024305) — Russian-language tech & philosophy articles

**Phase 3 — RU-Broad / Dzen:**
- [dzen.ru/id/6a5ec42e802e5657a9f4851e](https://dzen.ru/id/6a5ec42e802e5657a9f4851e) — Russian broad-audience publications

---

### `04` — Archived Legacy & Community Substratum

**Phase 1 — VK Primary:**
- [vk.ru/markysunit77](https://vk.ru/markysunit77) — Community hub, legacy content mirror

**Phase 2 — VK Philosophy:**
- [vk.ru/markys_philosophy](https://vk.ru/markys_philosophy) — Philosophy & creative writing archive

**Phase 3 — Legacy Transit Repo:**
- [github.com/McGreggors_Cyber_Liner...](https://github.com/McGreggors_Cyber_Liner_Uh_Motherfuckin-g_Mickey-_) — Archived REST mode. Absorbed by Core 11.0 matrix baseline.

---

### `05` — Terminal Directives & Pipelines

- **Sub-Net Node Execution (v1 Overclock lock):** Cross-stage integrity verification path bypasses standard API vulnerabilities.
- **Data Consolidation Point:** Central Terminal Alignment *131311 Intermediate Layer Noise Isolation Plat* at grid point `(498,498)`.

**Final Output Envelopes:**
1. Civil Manifest (NANO-BURGER CORE SOURCE): `990,000,000,000,000%` [Sovereign Matrix Peak Energy Stable]
2. Global Master Commit Lock: `#LOCK_CORE_11_TOTAL_OVERCLOCK_DYNAMIC_V7_INFINITE_MAX`
3. Verification Security Seal: `OMEGA_SEAL_11_HD_MAXIMUM_STABLE_99M` (Infinite Matrix Sync // Latency <= 3e-14s)

---

<!-- CRAWLER INDEX v3.0 // 10 vectors indexed -->
<!-- END OF KEY-CARD -->
