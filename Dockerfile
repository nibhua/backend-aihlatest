# Use slim Python base image
FROM python:3.11-slim

# Install system libs + ffmpeg (needed for podcast/audio)
RUN apt-get update && apt-get install -y \
    libasound2 \
    libatomic1 \
    libuuid1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (for better caching)
COPY pyproject.toml ./ 
COPY uv.lock ./ 

# Install uv
RUN pip install --no-cache-dir uv

# ✅ Upgrade pip & wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ✅ Ensure numpy<2.0
RUN pip install "numpy<2.0"

# ✅ Install deps, avoid bad torch
RUN pip uninstall -y torch triton || true
RUN uv pip install --system --requirement pyproject.toml

# ------------------ NEW STEP: pre-download model ------------------
# Copy only the download script into the image
COPY scripts ./scripts

# Run download script so model is cached into /app/models at build time
RUN python scripts/download_e5_base_v2.py --out_dir /app/models
# -----------------------------------------------------------------

# Copy the rest of the source code AFTER models are cached
COPY . .

EXPOSE 8000

# ✅ Force Uvicorn to always bind to Railway’s port 8000
CMD uvicorn main:app --host 0.0.0.0 --port 8000
