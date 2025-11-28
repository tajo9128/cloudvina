# Quick Start: GitHub Codespaces

**Fastest way to test BioDockify in the cloud (no local setup needed!)**

## Step 1: Create GitHub Account
Go to [github.com](https://github.com) and sign up (free)

## Step 2: Create Repository

1. Click **+** → **New repository**
2. Name: `BioDockify`
3. Visibility: Public
4. Click **Create repository**

## Step 3: Push Code

Copy and run these commands in PowerShell:

```powershell
cd c:\Users\tajo9\.gemini\antigravity\playground\exo-pinwheel\BioDockify

# Configure Git (one-time setup)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Commit
git commit -m "Phase 1: Docker container for AutoDock Vina"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/BioDockify.git

# Push
git branch -M main
git push -u origin main
```

## Step 4: Launch Codespace

1. Go to your repo: `github.com/YOUR_USERNAME/BioDockify`
2. Click green **Code** button → **Codespaces** tab
3. Click **Create codespace on main**
4. Wait 2 minutes for setup

## Step 5: Test Docker Container

Once VS Code loads in your browser, run in the terminal:

```bash
cd docker

# Build (takes 3-5 mins first time)
docker build -t BioDockify:latest .

# Create test files
cd test_data
cat > test_receptor.pdb << 'EOF'
HEADER    TEST PROTEIN
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  CA  ALA A   4       3.000   0.000   0.000  1.00  0.00           C
END
EOF

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

# Run test (1-5 minutes)
docker run --rm \
  -v $(pwd)/test_data:/app/work \
  BioDockify:latest \
  python test_local.py
```

**Success looks like:**
```
============================================================
✅ LOCAL TEST COMPLETE!
============================================================

Docker container is working correctly! ✓
```

## Done! 🎉

Your Docker container is validated and ready for AWS deployment.

**Next Steps:**
- See `CODESPACES_SETUP.md` for detailed guide
- See `AWS_SETUP.md` when ready to deploy

**Free Tier:** 60 hours/month Codespaces (plenty for testing!)
