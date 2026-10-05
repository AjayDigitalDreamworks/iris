# IRIS backend. CPU-only by default so the image runs anywhere; a CUDA base
# image can be swapped in without changing application code.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    IRIS_MODEL_DIR=/app/models \
    IRIS_DATA_DIR=/app/data

# libGL/libglib are required by opencv-python; ffmpeg is used by video ingest.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/
COPY scripts/ /app/scripts/
COPY models/ /app/models/
COPY data/sample_images/ /app/data/sample_images/
COPY data/sample_videos/ /app/data/sample_videos/

RUN useradd --create-home --uid 10001 iris \
    && mkdir -p /app/data /app/logs /app/storage \
    && chown -R iris:iris /app
USER iris

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
