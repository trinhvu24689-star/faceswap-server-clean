FROM python:3.10.11 AS builder

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN python -m venv .venv

COPY requirements.txt .
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.10.11-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 libglib2.0-dev libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv .venv/

# 👉 THAY ĐOẠN COPY CŨ BẰNG 2 DÒNG NÀY
COPY . .
RUN mkdir -p models

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]