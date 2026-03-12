# Use a slim Python image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for sklearn/xgboost/lightgbm)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        gfortran \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose port used by FastAPI
EXPOSE 8000

# Default command: run the FastAPI app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

