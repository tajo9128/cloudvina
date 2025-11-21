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
RUN pip install --no-cache-dir boto3==1.34.0

# Copy runner script
COPY run_docking.py /app/run_docking.py
WORKDIR /app

# Verify installation
RUN vina --version && obabel --version

# Entry point
ENTRYPOINT ["python", "run_docking.py"]
