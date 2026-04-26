FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    accelerate \
    "transformers>=4.45.0" \
    "tokenizers>=0.22.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "httpx>=0.27.0" \
    "pydantic>=2.0" \
    "regex>=2024.0.0" \
    "openenv-core @ git+https://github.com/meta-pytorch/OpenEnv.git"

COPY . /app

# The Space repo contains the whole meta-hackathon repo, so the env code lives
# under /app/PromptWar_env. Sanity-check at build time so we fail fast.
ENV PYTHONPATH=/app
ENV PROMPTWAR_LOAD_CONSUMER_MODEL=0

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
