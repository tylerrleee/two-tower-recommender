FROM python:3.11-slim

# everything runs through this workdir
WORKDIR /app 

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Define container's dir structure
COPY src/ /app/src/
COPY api/ /app/api/
COPY config/ /app/config/
COPY models/ /app/models/
COPY database/ /app/database/


# env setup
ENV MODEL_PATH=/app/models/best_model.pt \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

  
EXPOSE 8000

# NON-ROOT
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"


CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
