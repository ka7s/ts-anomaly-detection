FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cache-friendly)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src /app/src

# Copy application code
COPY src /app/src
COPY configs /app/configs

# Make src importable
ENV PYTHONPATH=/app/src

# Default env vars (override at runtime)
ENV TSAD_DATA_ROOT=/data
ENV TSAD_REPORTS_DIR=/app/reports

EXPOSE 8000

CMD ["uvicorn", "tsad.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
