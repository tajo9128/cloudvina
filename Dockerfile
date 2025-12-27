# Simplified Dockerfile for HuggingFace Spaces Free Tier
FROM python:3.10-slim

# Install system dependencies (including RDKit requirements AND OpenBabel for ODDT)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    swig \
    libopenbabel-dev \
    libopenbabel7 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Setup User (required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements first for caching
COPY --chown=user requirements.txt .

# Install dependencies (Force Numpy < 2.0 for ODDT stability if needed, though git version is usually fine)
# Install ODDT manually via git to get latest fixes and use --no-build-isolation
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/oddt/oddt.git --no-build-isolation

# Copy app code
COPY --chown=user . .

# HuggingFace Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
