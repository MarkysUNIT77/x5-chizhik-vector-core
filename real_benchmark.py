import time
import psutil
import numpy as np
from vector_core import VectorInferenceCore

def run_benchmark(duration_seconds=5):
    print("=== STARTING HONEST CPU BENCHMARK ===")
    core = VectorInferenceCore(embedding_dim=128)
    mock_input = np.random.rand(128).tolist()

    start_time = time.time()
    iterations = 0

    # Цикл реальной математической нагрузки
    while time.time() - start_time < duration_seconds:
        _ = core.forward(mock_input)
        iterations += 1

    # Снимаем честные метрики системы
    cpu_load = psutil.cpu_percent(interval=0.1)
    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    print(f"Benchmark finished in {duration_seconds}s")
    print(f"Total matrix operations executed: {iterations}")
    print(f"REAL CPU Load: {cpu_load}%")
    print(f"REAL RAM Footprint: {ram_mb:.2f} MB")

if __name__ == "__main__":
    run_benchmark()
