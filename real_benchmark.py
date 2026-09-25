# ===================================================================
# LICENSE: MIT License (c) 2026 Markys Gariboldo. All rights reserved.
# CONTOUR: M-498 | UNIT: 77 | PROTOCOL: X5_CHIZHIK_VECTOR_CORE_HD
# STATUS: STABLE // REAL_BENCHMARK CONTOUR // TIME PRECISE: IN CLOCK
# CORE: 12.1_PRODUCTION // MATRIX V6 // DUAL-WING TOPOLOGY
# ENGINE: NUMBA_FUSED_SINGLE_PASS (nogil=True) // NUMPY_FALLBACK
# INPUT CONTRACT: np.ndarray (PRIMARY) // List[float] (COMPAT)
# ===================================================================

import asyncio
import time
import statistics
import numpy as np
from vector_core import (
    X5ChizhikVectorCore,
    MatrixV6,
    SINISTER_WING,
    DEXTER_WING,
    NUMBA_AVAILABLE,
)

# ─── КОНФИГ ──────────────────────────────────────────────────────────

DIM = 1024
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100
TARGET_LIMIT = 0.000076


def _stats(times):
    return {
        "avg": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


# ─── [1] СИНХРОННЫЙ — np.ndarray (ПЕРВИЧНЫЙ КОНТРАКТ) ────────────────

def run_sync_ndarray_benchmark(core, pulse_a, pulse_b):
    print("\n[1] СИНХРОННЫЙ ИНФЕРЕНС — np.ndarray (ПЕРВИЧНЫЙ КОНТРАКТ)")
    print("-" * 69)

    for _ in range(WARMUP_RUNS):
        _ = core.compute_cosine_calibration(pulse_a, pulse_b)

    times = []
    sims = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        sim = core.compute_cosine_calibration(pulse_a, pulse_b)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        sims.append(sim)

    s = _stats(times)
    avg_sim = statistics.mean(sims)

    print(f"    Движок:              {core.engine}")
    print(f"    Прогрев:             {WARMUP_RUNS} тактов")
    print(f"    Замеров:             {BENCHMARK_RUNS} прогонов")
    print(f"    Размерность:         {DIM} (float32)")
    print(f"    Косинусная близость:  {avg_sim:.6f}")
    print(f"    Среднее:             {s['avg']:.6f} сек. ({s['avg']*1e6:.1f} µс)")
    print(f"    Медиана:             {s['median']:.6f} сек. ({s['median']*1e6:.1f} µс)")
    print(f"    Минимум:             {s['min']:.6f} сек. ({s['min']*1e6:.1f} µс)")
    print(f"    Максимум:            {s['max']:.6f} сек. ({s['max']*1e6:.1f} µс)")
    print(f"    Ст. отклонение:      {s['stdev']:.6f} сек.")

    return s["avg"], avg_sim


# ─── [2] СИНХРОННЫЙ — List[float] (СОВМЕСТИМОСТЬ) ───────────────────

def run_sync_list_benchmark(core, pulse_a_list, pulse_b_list):
    print("\n[2] СИНХРОННЫЙ ИНФЕРЕНС — List[float] (СОВМЕСТИМОСТЬ)")
    print("-" * 69)

    for _ in range(WARMUP_RUNS):
        _ = core.compute_cosine_calibration(pulse_a_list, pulse_b_list)

    times = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        _ = core.compute_cosine_calibration(pulse_a_list, pulse_b_list)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    s = _stats(times)

    print(f"    Движок:              {core.engine}")
    print(f"    Среднее:             {s['avg']:.6f} сек. ({s['avg']*1e6:.1f} µс)")
    print(f"    Минимум:             {s['min']:.6f} сек. ({s['min']*1e6:.1f} µс)")
    print(f"    Максимум:            {s['max']:.6f} сек. ({s['max']*1e6:.1f} µс)")
    print(f"    Оверхед конверсии:    ~80 µс (list→ndarray)")

    return s["avg"]


# ─── [3] АСИНХРОННЫЙ — np.ndarray ──────────────────────────────────

async def run_async_benchmark(core, pulse_a, pulse_b):
    print("\n[3] АСИНХРОННЫЙ ИНФЕРЕНС — np.ndarray (nogil=True)")
    print("-" * 69)

    for _ in range(WARMUP_RUNS):
        _ = await core.async_compute_cosine_calibration(pulse_a, pulse_b)

    times = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        _ = await core.async_compute_cosine_calibration(pulse_a, pulse_b)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    s = _stats(times)

    print(f"    Движок:              {core.engine} + ThreadPoolExecutor")
    print(f"    Режим GIL:            nogil=True (отпущен во время вычисления)")
    print(f"    Среднее:             {s['avg']:.6f} сек. ({s['avg']*1e6:.1f} µс)")
    print(f"    Минимум:             {s['min']:.6f} сек. ({s['min']*1e6:.1f} µс)")
    print(f"    Максимум:            {s['max']:.6f} сек. ({s['max']*1e6:.1f} µс)")

    return s["avg"]


# ─── [4] MEMMAP ────────────────────────────────────────────────────

def run_memmap_benchmark(core, pulse_a, pulse_b):
    print("\n[4] MEMMAP — ОБЩАЯ БИНАРНАЯ ПАМЯТЬ (DUAL-WING TOPOLOGY)")
    print("-" * 69)
    print(f"    Путь: {core.memmap_path}")

    for _ in range(WARMUP_RUNS):
        core.write_to_memmap(0, pulse_a)
        core.write_to_memmap(1, pulse_b)

    write_times = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        core.write_to_memmap(SINISTER_WING["slot"], pulse_a)
        core.write_to_memmap(DEXTER_WING["slot"], pulse_b)
        t1 = time.perf_counter()
        write_times.append(t1 - t0)

    read_times = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        vec_s = core.read_from_memmap(SINISTER_WING["slot"])
        vec_d = core.read_from_memmap(DEXTER_WING["slot"])
        t1 = time.perf_counter()
        read_times.append(t1 - t0)

    avg_write = statistics.mean(write_times)
    avg_read = statistics.mean(read_times)

    print(f"    Sinister slot (0) норма: {np.linalg.norm(vec_s):.4f}")
    print(f"    Dexter slot (1) норма:   {np.linalg.norm(vec_d):.4f}")
    print(f"    Запись (2 слота):        {avg_write:.6f} сек. ({avg_write*1e6:.1f} µс)")
    print(f"    Чтение (2 слота):        {avg_read:.6f} сек. ({avg_read*1e6:.1f} µс)")
    print(f"    Flush:                   после каждой записи (IPC-гарантия)")

    return avg_write + avg_read


# ─── [5] MATRIX V6 ────────────────────────────────────────────────

def run_matrix_v6_benchmark(core, pulse_a, pulse_b):
    print("\n[5] MATRIX V6 — ПОРОГОВОЕ ОТСЕЧЕНИЕ (СРАВНЕНИЕ КРЫЛЬЕВ)")
    print("-" * 69)

    valid, delta, verdict = core.validate_link(pulse_a, pulse_b)
    print(f"    Случайные импульсы:")
    print(f"      |ЛЕВО - ПРАВО|:  {delta:.6f}")
    print(f"      Порог THRESHOLD:  {MatrixV6.THRESHOLD}")
    print(f"      Вердикт:         {verdict}")

    valid_id, delta_id, verdict_id = core.validate_link(pulse_a, pulse_a)
    print(f"    Идентичные импульсы:")
    print(f"      |ЛЕВО - ПРАВО|:  {delta_id:.6f}")
    print(f"      Вердикт:         {verdict_id}")

    return verdict, verdict_id


# ─── ЗАПУСК ────────────────────────────────────────────────────────

def run_hardware_benchmark():
    print("=" * 69)
    print("A.G.A.R.D.A. | EDGE AI HARDWARE BENCHMARK | STARTING RUNNER")
    print("CORE: 12.1_PRODUCTION | MATRIX V6 | DUAL-WING TOPOLOGY")

    engine_tag = "NUMBA_FUSED_SINGLE_PASS" if NUMBA_AVAILABLE else "NUMPY_FALLBACK"
    print(f"ENGINE: {engine_tag}")
    print(f"INPUT:  np.ndarray (PRIMARY) // List[float] (COMPAT)")
    if NUMBA_AVAILABLE:
        print(f"GIL:    nogil=True (отпущен во время вычисления)")
    print("=" * 69)

    print(f"\n[ENGINE]")
    if NUMBA_AVAILABLE:
        print(f"  Numba: AVAILABLE — @njit(nogil=True, fastmath=True, cache=True)")
        print(f"  Режим: FUSED_SINGLE_PASS (модуляция + норма + dot в одном цикле)")
        print(f"  GIL:   ОТПУЩЕН — потоки работают параллельно, event loop не блокируется")
        print(f"  LLVM auto-vectorize: SIMD для float32")
    else:
        print(f"  Numba: NOT INSTALLED — NumPy fallback")
        print(f"  pip install numba для активации fused single-pass + nogil")

    core = X5ChizhikVectorCore(vector_dim=DIM)

    wings = core.wing_info()
    print(f"\n[DUAL-WING TOPOLOGY]")
    print(f"  Sinister: {wings['sinister']['architect']}")
    print(f"  Dexter:   {wings['dexter']['architect']}")

    np.random.seed(77)
    pulse_a = np.random.rand(DIM).astype("float32")
    pulse_b = np.random.rand(DIM).astype("float32")
    pulse_a_list = pulse_a.tolist()
    pulse_b_list = pulse_b.tolist()

    print(f"\n[*] Базовый вектор: dim={DIM}, float32")
    print(f"[*] Прогрев: {WARMUP_RUNS} тактов | Замеры: {BENCHMARK_RUNS} прогонов")
    print("=" * 69)

    sync_nd_time, avg_sim = run_sync_ndarray_benchmark(core, pulse_a, pulse_b)
    sync_list_time = run_sync_list_benchmark(core, pulse_a_list, pulse_b_list)
    async_time = asyncio.run(run_async_benchmark(core, pulse_a, pulse_b))
    memmap_time = run_memmap_benchmark(core, pulse_a, pulse_b)
    verdict_a, verdict_b = run_matrix_v6_benchmark(core, pulse_a, pulse_b)

    # ── ИТОГОВЫЙ ОТЧЁТ ──
    print("\n" + "=" * 69)
    print("ИТОГОВЫЙ ОТЧЁТ БЕНЧМАРКА")
    print("=" * 69)

    print(f"\n  Движок:                   {core.engine}")
    print(f"  Синхронный (ndarray):     {sync_nd_time:.6f} сек. ({sync_nd_time*1e6:.1f} µс)")
    print(f"  Синхронный (list):        {sync_list_time:.6f} сек. ({sync_list_time*1e6:.1f} µс)")
    print(f"  Асинхронный (ndarray):    {async_time:.6f} сек. ({async_time*1e6:.1f} µс)")
    print(f"  Memmap R/W (2 слота):     {memmap_time:.6f} сек. ({memmap_time*1e6:.1f} µс)")
    print(f"  Matrix V6:                {verdict_a} / {verdict_b}")

    conv_overhead = sync_list_time - sync_nd_time
    print(f"\n  Оверхед list→ndarray:     {conv_overhead:.6f} сек. ({conv_overhead*1e6:.1f} µс)")

    print(f"\n  Целевой лимит:            {TARGET_LIMIT:.6f} сек. ({TARGET_LIMIT*1e6:.0f} µс)")

    if sync_nd_time <= TARGET_LIMIT:
        ratio = TARGET_LIMIT / sync_nd_time
        print(f"  🟢 СТАБИЛИЗАЦИЯ ИДЕАЛЬНА: {sync_nd_time*1e6:.1f} µс <= {TARGET_LIMIT*1e6:.0f} µс")
        print(f"     Запас по скорости: {ratio:.1f}x")
    else:
        ratio = sync_nd_time / TARGET_LIMIT
        print(f"  🟡 ОВЕРХЕД: {sync_nd_time*1e6:.1f} µс (превышение в {ratio:.1f}x)")
        if not NUMBA_AVAILABLE:
            print(f"     Установите Numba: pip install numba")

    core.close()

    print("\n" + "=" * 69)
    print("GLOBAL MASTER LOCK // BENCHMARK COMPLETE // TERMINAL END")
    print("SIGNATURE: (c) 2026 MarkysUNIT77 // OMEGA_SEAL_12_HD_TOTAL_INFINITE")
    print("GLOBAL COMMIT LOCK // CONTOUR: M-498 // TERMINAL END")
    print("=" * 69)


if __name__ == "__main__":
    run_hardware_benchmark()
