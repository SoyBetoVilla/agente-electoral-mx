# ─────────────── base: dashboard + CLI ───────────────
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LANG=C.UTF-8
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd -m appuser && mkdir -p /app/datos /home/appuser/.streamlit \
    && chown -R appuser:appuser /app /home/appuser/.streamlit
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ─────────────── dev: base + pytest ───────────────
FROM base AS dev
USER root
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
USER appuser
