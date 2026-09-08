# Phase 12 — Django backend container.
# Postgres + Django are containerized here; Hadoop/Kafka/Spark continue
# running natively in WSL2 (see docker-compose.yml comments for why).

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# psycopg2 (non-binary builds) needs libpq headers; using psycopg2-binary
# in requirements.txt avoids that, but keep this for portability if it's
# ever swapped to the non-binary package.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Applies Django's own migrations (auth/sessions/admin/authtoken only —
# the fraud_api models are managed=False and are correctly skipped, same
# as running this natively) then starts the dev server bound to all
# interfaces so it's reachable from outside the container.
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]