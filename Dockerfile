# Base image with common dependencies
FROM python:3.9-slim as base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Data Collection image
FROM base as data-collection
COPY pipeline.py .
ENTRYPOINT ["python", "pipeline.py"]

# Data Storage image
FROM base as data-storage
COPY mongo.py .
ENTRYPOINT ["python", "mongo.py"]

# Analysis image
FROM base as analysis
COPY topk_analyzer.py .
ENTRYPOINT ["python", "topk_analyzer.py"]