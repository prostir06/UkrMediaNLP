# UkrMediaNLP — multi-stage Streamlit image
ARG PRELOAD_MODELS=false

FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU torch comes from requirements.txt (--extra-index-url).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

ARG PRELOAD_MODELS=false
ARG SPACY_MODEL=uk_core_news_sm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/app/.cache/huggingface \
    PRELOAD_MODELS=${PRELOAD_MODELS} \
    SPACY_MODEL=${SPACY_MODEL}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .
COPY scripts/docker_entrypoint.sh /usr/local/bin/docker_entrypoint.sh

RUN chmod +x /usr/local/bin/docker_entrypoint.sh \
    && mkdir -p /app/.cache/huggingface /app/.cache/articles \
    && chown -R appuser:appuser /app \
    && python -m spacy download ${SPACY_MODEL} || true

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

ENTRYPOINT ["docker_entrypoint.sh"]
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
