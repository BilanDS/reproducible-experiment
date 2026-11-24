FROM python:3.10-slim

WORKDIR /app

# Встановлення системних утиліт
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Інсталяція Python-залежностей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіювання коду експерименту
COPY train.py .

CMD ["python", "train.py"]