FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
# gcc and python3-dev needed for some pip packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Dependencies (Optimized for CPU/Free Tier)
# Use --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy Code
COPY . .

# Run FastAPI on port 7860 (Required for Hugging Face Spaces)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
