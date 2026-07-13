# ====================================================================================
# McGreggors Cyber Liner // x5-chizhik-vector-core // Dockerfile
# CONTAINER TYPE: RIDA-CAPSULE // ULTRA-LIGHTWEIGHT DEPLOYMENT
# ====================================================================================

FROM python:3.10-slim

# Настройка окружения под глухой вакуум 131311
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Установка голого NumPy без корпоративного оверхеда
RUN pip install --no-cache-dir numpy==1.26.4

# Сливаем векторизованное ядро напрямую в сектор
COPY vector_core.py /app/
COPY real_benchmark.py /app/

# Исполняемый триггер при запуске контейнера на ТСД
CMD ["python", "real_benchmark.py"]
