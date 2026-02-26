FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY meridian/ meridian/
COPY scripts/ scripts/
RUN chmod +x scripts/*.sh

RUN pip install --no-cache-dir -e .

ENV MERIDIAN_DATA_DIR=/data

EXPOSE 7891

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
