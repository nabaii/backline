FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY data/ data/
COPY backend_api.py .
COPY main.py .

EXPOSE 8080

CMD ["gunicorn", "backend.backend_api:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120"]
