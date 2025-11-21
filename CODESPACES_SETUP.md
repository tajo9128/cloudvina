# GitHub Codespaces Setup for CloudVina Testing

GitHub Codespaces provides a cloud-based development environment with Docker pre-installed. Perfect for testing CloudVina without local installation!

## Step 1: Push Your Code to GitHub

### 1a. Create a GitHub Repository

1. Go to [github.com](https://github.com) (create account if needed - it's free!)
2. Click the **+** icon → **New repository**
3. Settings:
   - **Name**: `cloudvina`
   - **Description**: "Web-based molecular docking platform using AutoDock Vina"
   - **Visibility**: Public (or Private if you prefer)
   - **Don't** initialize with README (we already have one)
4. Click **Create repository**

### 1b. Initialize Git and Push

Open PowerShell in your cloudvina directory:

```powershell
cd c:\Users\tajo9\.gemini\antigravity\playground\exo-pinwheel\cloudvina

# Initialize Git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Phase 1 Docker container"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/cloudvina.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Note**: If this is your first time using Git, you may need to configure it:
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 2: Create a Codespace

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/cloudvina`
2. Click the green **Code** button
3. Click the **Codespaces** tab
4. Click **Create codespace on main**

**Wait 1-2 minutes** for your environment to spin up. You'll get:
- VS Code in your browser
- Ubuntu Linux environment
- Docker pre-installed
- 60 hours/month free (plenty for testing!)

## Step 3: Test the Docker Container

Once your Codespace loads, open the terminal (it's already open at the bottom) and run:

### Build the Docker image

```bash
cd docker
docker build -t cloudvina:latest .
```

This will take 3-5 minutes the first time.

### Create test files

```bash
cd test_data

# Create minimal test receptor
cat > test_receptor.pdb << 'EOF'
HEADER    TEST PROTEIN
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  CA  ALA A   4       3.000   0.000   0.000  1.00  0.00           C
END
EOF

# Create minimal test ligand
cat > test_ligand.sdf << 'EOF'
test_ligand
  OpenBabel

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.8660    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000   -0.8660    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  4  1  0  0  0  0
M  END
$$$$
EOF

cd ..
```

### Run the test

```bash
docker run --rm \
  -v $(pwd)/test_data:/app/work \
  cloudvina:latest \
  python test_local.py
```

**Expected output:**
```
============================================================
CloudVina - Local Test Mode
============================================================

[1/4] Converting ligand to PDBQT...
   ✓ Ligand converted
[2/4] Preparing receptor...
   ✓ Receptor prepared
[3/4] Running AutoDock Vina...
   (This may take 1-5 minutes...)
   ✓ Docking complete!
[4/4] Checking results...
   ✓ Output saved: /app/work/output.pdbqt
   ✓ Log saved: /app/work/log.txt

   📊 Best binding affinity: -X.X kcal/mol

============================================================
✅ LOCAL TEST COMPLETE!
============================================================
```

## Step 4: Verify Results

Check that output files were created:

```bash
ls -lh test_data/
```

You should see:
- `output.pdbqt` - Docked poses
- `log.txt` - Vina log with binding energies
- `ligand.pdbqt` - Converted ligand
- `receptor.pdbqt` - Prepared receptor

View the log:
```bash
cat test_data/log.txt
```

## Step 5: Test with Real Protein (Optional)

Download a real protein structure:

```bash
cd test_data

# Download HIV protease
wget https://files.rcsb.org/download/1HVR.pdb -O hiv_protease.pdb

# Extract the ligand from the protein
# (In a real workflow, you'd prepare this separately)

# Run docking with the real protein
# (You'll need to provide a separate ligand file)
```

## Troubleshooting

### "Cannot connect to Docker daemon"
- Codespaces has Docker pre-installed, but sometimes it takes a moment to start
- Wait 30 seconds and try again
- Or run: `sudo service docker start`

### "Out of space"
- Codespaces free tier has 32GB storage
- Clean up: `docker system prune -a`

### Build takes too long
- First build takes 3-5 minutes (downloading Vina, OpenBabel)
- Subsequent builds are cached and much faster

## Cost & Limits

**GitHub Codespaces Free Tier:**
- 60 hours/month of 2-core machine
- 15 GB/month of storage
- Perfect for testing and development!

**Upgrading:**
- $0.18/hour for 2-core (if you exceed free tier)
- You can pause/delete codespaces when not in use

## Next Steps After Testing Works

1. ✅ Docker container validated
2. Push to AWS ECR (from Codespaces!)
3. Test on AWS Batch
4. Begin Phase 2 (FastAPI backend)

## Quick Commands Reference

```bash
# Build image
docker build -t cloudvina:latest .

# Run test
docker run --rm -v $(pwd)/test_data:/app/work cloudvina:latest python test_local.py

# Check Docker status
docker ps
docker images

# Clean up
docker system prune -a
```

---

**Ready to test in the cloud!** 🚀
