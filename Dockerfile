# === Шаг 1: Сборочный контейнер (скомпилирует psutil) ===
FROM python:3.10-alpine AS builder

# Устанавливаем ВСЕ необходимые инструменты компиляции под Alpine
RUN apk add --no-cache gcc musl-dev linux-headers python3-dev

WORKDIR /app

# Скачиваем и собираем wheel-пакеты в отдельную директорию
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/app/wheels numpy==1.26.4 psutil==5.9.8

# === Шаг 2: Финальный чистый контейнер (минимальный вес) ===
FROM python:3.10-alpine

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Копируем уже скомпилированные на первом шаге бинарные пакеты
COPY --from=builder /app/wheels /app/wheels

# Устанавливаем их моментально без повторной компиляции
RUN pip install --upgrade pip && \
    pip install --no-cache-dir /app/wheels/* && \
    rm -rf /app/wheels

# Копируем наши скрипты
COPY vector_core.py real_benchmark.py ./

ENTRYPOINT ["python", "real_benchmark.py"]
