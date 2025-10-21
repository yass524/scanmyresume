# ---------- base ----------
FROM python:3.11-slim

# Make pip more resilient
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt .

# 1) Install CPU-only PyTorch first to avoid CUDA (massive) downloads
#    Keep the version compatible with sentence-transformers/transformers.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1

# 2) Install the rest (with longer timeout & retries)
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

# Copy app code
COPY . .

# Expose / run
# Use the PORT provided by Cloud Run (defaults to 8080 locally)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
