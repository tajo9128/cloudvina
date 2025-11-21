# Docker Installation Guide for Windows

Docker is required to build and test the CloudVina container locally.

## Step 1: Download Docker Desktop

1. Go to [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Click "Download for Windows"
3. Run the installer (`Docker Desktop Installer.exe`)

## Step 2: Install Docker Desktop

1. Follow the installation wizard
2. **Important**: Enable WSL 2 if prompted (Windows Subsystem for Linux)
3. Restart your computer when installation completes

## Step 3: Verify Installation

Open PowerShell and run:

```powershell
docker --version
docker info
```

You should see output like:
```
Docker version 24.0.x, build xxxxx
```

## Step 4: Test Docker

Run the hello-world container:

```powershell
docker run hello-world
```

If you see "Hello from Docker!", you're ready to go!

## Troubleshooting

### "WSL 2 installation is incomplete"

1. Open PowerShell as Administrator
2. Run:
   ```powershell
   wsl --install
   ```
3. Restart computer

### "Docker daemon is not running"

1. Start Docker Desktop from Start Menu
2. Wait for it to fully start (whale icon in system tray should be stable)

### "Hardware virtualization is not enabled"

1. Restart computer and enter BIOS (usually F2, F10, or Del during boot)
2. Enable "Intel VT-x" or "AMD-V" under CPU settings
3. Save and exit BIOS

## Alternative: Use Cloud Testing

If you cannot install Docker locally, you can:

1. **Use GitHub Codespaces** (has Docker pre-installed)
2. **Use AWS Cloud9** (cloud-based IDE with Docker)
3. **Skip to AWS testing** (deploy directly to AWS Batch)

---

**Once Docker is installed, come back and we'll test the CloudVina container!**
