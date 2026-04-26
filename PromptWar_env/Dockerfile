FROM huggingface/transformers-pytorch-gpu

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir \
    accelerate \
    transformers>=4.45.0 \
    fastapi>=0.115.0 \
    uvicorn>=0.24.0 \
    httpx>=0.27.0 \
    pydantic>=2.0 \
    regex>=2024.0.0 \
    tokenizers>=0.22.0 \
    "openenv-core @ git+https://github.com/meta-pytorch/OpenEnv.git"

ENV PYTHONPATH=/app

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
