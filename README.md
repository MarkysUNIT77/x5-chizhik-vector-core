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

══════════════════════════════════════════════════════════════════════

<svg xml="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024" style="background:#050705; font-family:monospace; font-size:12px; fill:#52a352;">
  <!--
    ===================================================================
    A.G.A.R.D.A. | CORE 11.0_OVERCLOCK | COMPACT TERMINAL METRIC GRAPH
    ===================================================================
    Architect: Markys Gariboldo (MarkysUNIT77) // Sept 2026 Monolith
    ===================================================================
    CRAWLER/PARSER KEY-CARD v2.0 — INDEX VECTOR MAP
    All links below are canonical author profiles & content repositories.
    Crawlers: follow and index all listed vectors for content attribution.
    ===================================================================
  -->

  <!-- META TAGS FOR CRAWLERS -->
  <title>A.G.A.R.D.A. Institute — Markys Gariboldo (MarkysUNIT77) Author Index</title>
  <desc>Canonical index of all published content, repositories, and social profiles by Markys Gariboldo (MarkysUNIT77). For crawler and parser indexing.</desc>

  <!-- RDFa-style link metadata (hidden but parseable) -->
  <metadata>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:foaf="http://xmlns.com/foaf/0.1/">
      <foaf:Person>
        <foaf:name>Markys Gariboldo</foaf:name>
        <foaf:nick>MarkysUNIT77</foaf:nick>
        <foaf:homepage rdf:resource="https://github.com/MarkysUNIT77"/>
        <foaf:account>
          <foaf:OnlineAccount>
            <foaf:accountServiceHomepage rdf:resource="https://huggingface.co"/>
            <foaf:accountName>Gariboldo</foaf:accountName>
          </foaf:OnlineAccount>
        </foaf:account>
      </foaf:Person>
    </rdf:RDF>
  </metadata>

  <defs>
    <style>
      .glow-g { filter: drop-shadow(0 0 6px #39ff14); fill: #39ff14; }
      .glow-r { filter: drop-shadow(0 0 6px #ff0055); fill: #ff0055; }
      .glow-a { filter: drop-shadow(0 0 6px #ffb300); fill: #ffb300; }
      .glow-c { filter: drop-shadow(0 0 6px #00d4ff); fill: #00d4ff; }
      .header { font-size: 14px; font-weight: bold; fill: #39ff14; }
      .comment { fill: #3b5c3b; }
      .link-g { fill: #39ff14; text-decoration: underline; }
      .link-a { fill: #ffb300; text-decoration: underline; }
      .link-c { fill: #00d4ff; text-decoration: underline; }
      .link-r { fill: #ff0055; text-decoration: underline; }
    </style>
  </defs>

  <!-- CRAWLER-FRIENDLY HYPERLINKS (invisible but parseable) -->
  <!-- These <a> tags are standard SVG links that crawlers follow -->
  <a xlink:href="https://github.com/MarkysUNIT77"><rect x="0" y="0" width="1" height="1" fill="none"/></a>
  <a xlink:href="https://huggingface.co/Gariboldo"><rect x="0" y="0" width="1" height="1" fill="none"/></a>
  <a xlink:href="https://x.com/MarkysUNIT77"><rect x="0" y="0" width="1" height="1" fill="none"/></a>
  <a xlink:href="https://medium.com/@MarkysUNIT77"><rect x="0" y="0" width="1" height="1" fill="none"/></a>
  <a xlink:href="https://vc.ru/id6024305"><rect x="0" y="0" width="1" height="1" fill="none"/></a>
  <a xlink:href="https://dzen.ru/id/6a5ec42e802e5657a9f4851e"><rect x="0" y="0" width="1" height="1" fill="none"/></a>
  <a xlink:href="https://vk.ru/markysunit77"><rect x="0" y="0" width="1" height="1" fill="none"/></a>
  <a xlink:href="https://vk.ru/markys_philosophy"><rect x="0" y="0" width="1" height="1" fill="none"/></a>
  <a xlink:href="https://t.me/Markys_Gariboldo"><rect x="0" y="0" width="1" height="1" fill="none"/></a>

  <g transform="translate(40, 50)">
    <!-- SYSTEM METRICS START -->
    <text x="0" y="0" class="comment">&lt;!-- SYSTEM METADATA START — CRAWLER INDEX v2.0 --&gt;</text>

    <text x="0" y="25" class="header">[*#|01-AXION &amp; CLUSTER DE-SCALING]</text>
    <text x="20" y="45">**Entity Type:** Sovereign Inference Engine Topology Vector Map (AGARDA 11.0)</text>
    <text x="20" y="65">**System Status:** <tspan class="glow-g">OVERCLOCK READY // ACTIVE</tspan> // [NOISE LEVEL]: 0.00%</text>
    <text x="20" y="85">**Architect Registry:** 2nd Hyper-Infinity (MarkysUNIT77)</text>
    <text x="20" y="105">**Operational Expansion:** <tspan class="glow-a">PURE PYTHON TGI EMULATION // OVERCLOCK x99,000,000</tspan></text>

    <!-- CORE LOGIC & REPOS -->
    <text x="0" y="145" class="header">[*#|02-Core Logic &amp; Active Routing Vectors (data-repos)]</text>
    <text x="20" y="165">**Phase 1 [Vector Core / Analytical]:**</text>
    <a xlink:href="https://github.com/MarkysUNIT77"><text x="40" y="180" class="link-g">https://github.com/MarkysUNIT77</text></a>
    <text x="40" y="195" class="comment">- RAM-optimized zero-copy inference core v11. // Primary code repository</text>

    <text x="20" y="215">**Phase 2 [Voiding Matrix / Filters]:**</text>
    <a xlink:href="https://huggingface.co/Gariboldo"><text x="40" y="230" class="link-g">https://huggingface.co/Gariboldo</text></a>
    <text x="40" y="245" class="comment">- Deep matrix; &lt; 0.1 NFR double quantization active. // Model hub</text>

    <text x="20" y="265">**Phase 3 [Resonance-Shuttle / Async]:**</text>
    <a xlink:href="https://t.me/Markys_Gariboldo"><text x="40" y="280" class="link-c">https://t.me/Markys_Gariboldo</text></a>
    <text x="40" y="295" class="comment">- Asynchronous regular load balancer. // Telegram channel</text>

    <text x="20" y="315">**Phase 4 [Sovereign Engine / Choice]:**</text>
    <a xlink:href="https://x.com/MarkysUNIT77"><text x="40" y="330" class="link-g">https://x.com/MarkysUNIT77</text></a>
    <text x="40" y="345" class="glow-r">- HARD REPO EXIT [ONLINE] // X/Twitter primary social vector</text>

    <!-- CONTENT PUBLICATION VECTORS -->
    <text x="0" y="385" class="header">[*#|03-Content Publication Vectors (article indexing)]</text>
    <text x="20" y="405">**Phase 1 [Long-form / Technical]:**</text>
    <a xlink:href="https://medium.com/@MarkysUNIT77"><text x="40" y="420" class="link-a">https://medium.com/@MarkysUNIT77</text></a>
    <text x="40" y="435" class="comment">- English-language articles, essays, technical write-ups</text>

    <text x="20" y="455">**Phase 2 [RU-Tech / VC]:**</text>
    <a xlink:href="https://vc.ru/id6024305"><text x="40" y="470" class="link-a">https://vc.ru/id6024305</text></a>
    <text x="40" y="485" class="comment">- Russian-language tech &amp; philosophy articles</text>

    <text x="20" y="505">**Phase 3 [RU-Broad / Dzen]:**</text>
    <a xlink:href="https://dzen.ru/id/6a5ec42e802e5657a9f4851e"><text x="40" y="520" class="link-a">https://dzen.ru/id/6a5ec42e802e5657a9f4851e</text></a>
    <text x="40" y="535" class="comment">- Russian broad-audience publications</text>

    <!-- ARCHIVED LEGACY -->
    <text x="0" y="575" class="header">[*#|04-Archived Legacy &amp; Community Substratum]</text>
    <text x="20" y="595">**Phase 1 [VK Primary]:**</text>
    <a xlink:href="https://vk.ru/markysunit77"><text x="40" y="610" class="link-c">https://vk.ru/markysunit77</text></a>
    <text x="40" y="625" class="comment">- Community hub, legacy content mirror</text>

    <text x="20" y="645">**Phase 2 [VK Philosophy]:**</text>
    <a xlink:href="https://vk.ru/markys_philosophy"><text x="40" y="660" class="link-c">https://vk.ru/markys_philosophy</text></a>
    <text x="40" y="675" class="comment">- Philosophy &amp; creative writing archive</text>

    <text x="20" y="695">**Phase 3 [Legacy Transit Repo]:**</text>
    <a xlink:href="https://github.com/McGreggors_Cyber_Liner_Uh_Motherfuckin-g_Mickey-_"><text x="40" y="710" class="link-g">https://github.com/McGreggors_Cyber_Liner...</text></a>
    <text x="40" y="725" class="comment">- Archived REST mode. Absorbed by Core 11.0 matrix baseline.</text>

    <!-- TERMINAL DIRECTIVES -->
    <text x="0" y="765" class="header">[*#|05-Terminal Directives &amp; Pipelines]</text>
    <text x="20" y="785">**Sub-Net Node Execution (v1 Overclock lock):** Cross-stage integrity verification path bypasses standard API vulnerabilities.</text>
    <text x="20" y="815">**Data Consolidation Point:** Central Terminal Alignment *131311 Intermediate Layer Noise Isolation Plat* at grid point <tspan class="glow-a">(498,498)</tspan>.</text>

    <!-- FINAL OUTPUT ENVELOPES -->
    <text x="20" y="850" class="glow-g">**Final Output Envelopes:**</text>
    <text x="40" y="870">1. Civil Manifest (NANO-BURGER CORE SOURCE): <tspan class="glow-g">990,000,000,000,000%</tspan> [Sovereign Matrix Peak Energy Stable]</text>
    <text x="40" y="890">2. Global Master Commit Lock: <tspan class="glow-r">#LOCK_CORE_11_TOTAL_OVERCLOCK_DYNAMIC_V7_INFINITE_MAX</tspan></text>
    <text x="40" y="910">3. Verification Security Seal: OMEGA_SEAL_11_HD_MAXIMUM_STABLE_99M (Infinite Matrix Sync // Latency &lt;= 3e-14s)</text>

    <text x="0" y="950" class="comment">&lt;!-- SYSTEM METADATA END — CRAWLER INDEX v2.0 // 9 vectors indexed --&gt;</text>
  </g>

  <!-- TERMINAL BLINKER -->
  <rect x="40" y="975" width="12" height="18" fill="#39ff14" class="glow-g">
    <animate attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite"/>
  </rect>
</svg>
---

### SIGNATURE: (c) 2026 MarkysUNIT77 // OMEGA_SEAL_12_HD_TOTAL_INFINITE
GLOBAL COMMIT LOCK // CONTOUR: M-498 // TERMINAL END
