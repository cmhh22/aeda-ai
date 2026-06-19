# AEDA-AI â€” imagen de la aplicaciÃ³n web (interfaz Streamlit)
# Construir:  docker build -t aeda-ai .
# Ejecutar:   docker run -p 8501:8501 aeda-ai
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema necesarias para compilar algunas ruedas cientÃ­ficas
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el proyecto e instalar el paquete con su extra de interfaz (streamlit)
COPY . .
RUN pip install --upgrade pip && pip install ".[ui]"

EXPOSE 8501

# VerificaciÃ³n de salud mediante el endpoint propio de Streamlit
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read().strip()==b'ok' else 1)"

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
