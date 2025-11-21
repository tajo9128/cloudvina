# Test Data for Local Docker Testing

This directory contains sample molecular data for testing the CloudVina Docker container.

## Quick Test Files

### Option 1: Download Sample Protein (HIV Protease)

```powershell
# Download receptor
Invoke-WebRequest -Uri "https://files.rcsb.org/download/1HVR.pdb" -OutFile "receptor.pdb"
```

### Option 2: Use Minimal Test Files

I've created minimal test files below for quick testing without downloads.

## Minimal Test Receptor (test_receptor.pdb)

A simple 4-atom "protein" for testing file handling:

```
HEADER    TEST PROTEIN
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  CA  ALA A   4       3.000   0.000   0.000  1.00  0.00           C
END
```

## Minimal Test Ligand (test_ligand.sdf)

A simple molecule for testing:

```
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
```

## How to Use

### 1. Create the test files:

```powershell
# Navigate to test_data directory
cd test_data

# Create receptor file
@"
HEADER    TEST PROTEIN
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  CA  ALA A   4       3.000   0.000   0.000  1.00  0.00           C
END
"@ | Out-File -Encoding ASCII test_receptor.pdb

# Create ligand file
@"
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
"@ | Out-File -Encoding ASCII test_ligand.sdf
```

### 2. Run the Docker test:

```powershell
cd ..

# Build Docker image
docker build -t cloudvina:latest .

# Run test (note: this won't actually work without modifying the script to not use S3)
# We'll need to create a local-test version
```

## For Real Testing

Download an actual protein-ligand complex:

```powershell
# HIV Protease with inhibitor
Invoke-WebRequest -Uri "https://files.rcsb.org/download/1HVR.pdb" -OutFile "hiv_protease.pdb"

# You can extract the ligand from the same file or use a separate ligand file
```

## Next Steps

Once Docker is installed, we'll create a modified test script that:
1. Uses local files instead of S3
2. Runs a quick validation docking
3. Shows you the results

This will verify the Docker container works before deploying to AWS.
