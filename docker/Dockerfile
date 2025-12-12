# CloudVina - AutoDock Vina Docker Container
# Purpose: Run molecular docking in isolated, scalable environment
# Target: AWS Batch (Free Tier t2.micro initially, c6i.large Spot for production)

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    openbabel \
    python3-openbabel \
    && rm -rf /var/lib/apt/lists/*

# Install AutoDock Vina
# Using official release from Scripps Research
RUN wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 \
    && chmod +x vina_1.2.5_linux_x86_64 \
    && mv vina_1.2.5_linux_x86_64 /usr/local/bin/vina

# Install Python dependencies
# Keeping it lightweight: Only boto3 needed.
# rDock runs via binary, Vina runs via binary.
RUN pip install --no-cache-dir boto3==1.34.0

# Copy runner script
ARG CACHEBUST=1
COPY docker/main.py /app/main.py

# Copy API services (Docking Engine, etc.)
# We copy to /app/api/services so python can import it
# But import paths need to match.
# If we copy to /app/services, inside main.py we can `import services.docking_engine`
COPY api/services /app/services
WORKDIR /app

# SKIPPING rDock: Repository structure changed, build from source no longer works
# rDock would require significant rework to install. For now, using Vina + Gnina.
# TODO: Add rDock via conda or pre-built binary when available

# Install Gnina (v1.0.3) - AI-powered docking
# Downloading pre-compiled binary
RUN wget https://github.com/gnina/gnina/releases/download/v1.0.3/gnina \
    && chmod +x gnina \
    && mv gnina /usr/local/bin/gnina

# Verify installation (Vina and Gnina only)
RUN vina --version && obabel --version && gnina --version || echo "Gnina may need CUDA"

# Entry point
ENTRYPOINT ["python", "main.py"]
