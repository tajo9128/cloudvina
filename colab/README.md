# BioDockify Colab Worker - Quick Start Guide

This guide explains how to run MD simulations on Google Colab's free GPU using BioDockify.

---

## Prerequisites

- BioDockify account (free at https://biodockify.com)
- Google account (for Colab access)
- Upstash Redis URL (provided in your BioDockify dashboard)

---

## Setup Instructions

### Step 1: Open the Worker Notebook

1. Go to Google Colab: https://colab.research.google.com
2. Click **File → Upload notebook**
3. Upload [`BioDockify_Worker.ipynb`](./BioDockify_Worker.ipynb)
4. **Important:** Enable GPU runtime
   - Click **Runtime → Change runtime type**
   - Set Hardware accelerator to **GPU (T4)**
   - Click **Save**

### Step 2: Install Dependencies (Cell 1)

Run the first cell to install OpenMM, MDAnalysis, and PLIP:

```python
!pip install -q condacolab
import condacolab
condacolab.install()
!conda install -c conda-forge openmm mdanalysis -y
!pip install -q celery redis plip matplotlib pandas
```

**⏱️ Time:** ~5 minutes (one-time per session)

### Step 3: Configure Redis Connection (Cell 2)

Get your Redis URL from BioDockify dashboard:

1. Login to BioDockify
2. Go to **Settings → Worker Configuration**
3. Copy your Upstash Redis URL
4. Paste it in Cell 2:

```python
REDIS_URL = "rediss://default:YOUR_PASSWORD@your-endpoint.upstash.io:6379"
```

Run the cell. You should see:
```
✅ GPU Available: CUDA
✅ Configuration set!
```

### Step 4: Load Analysis Utilities (Cell 3)

Run Cell 3 to load trajectory analysis and interaction profiling functions.

### Step 5: Load Worker Implementation (Cell 4)

Run Cell 4 to define the OpenMM simulation worker with integrated analysis.

You should see:
```
✅ Worker implementation loaded with integrated analysis!
```

### Step 6: Start the Worker (Cell 5)

Run Cell 5 to start listening for jobs:

```python
celery_app.worker_main([...])
```

You should see:
```
🚀 Starting BioDockify Worker with Analysis...
📡 Listening for jobs from BioDockify...
⚠️  KEEP THIS CELL RUNNING to process jobs!
```

**⚠️ Important:** Keep this cell running! Closing the tab will stop the worker.

---

## Submitting a Job

### From BioDockify Web Interface:

1. Login to BioDockify
2. Navigate to **MD Simulation** (in main menu)
3. Upload your PDB file (protein structure)
4. Configure simulation:
   - Temperature: 300 K (default)
   - Simulation steps: 5000 (= ~10ps)
   - Forcefield: AMBER14 (recommended)
5. Click **Start Simulation**

The web interface will show real-time progress:
- Initializing → Loading → Energy Minimization → Equilibration → Simulating → Analyzing

---

## What Happens During Simulation

### Phase 1: MD Simulation (Progress 0-70%)

1. **Energy Minimization** - Removes steric clashes
2. **Equilibration** - 100 steps warm-up
3. **Production Run** - Main simulation (your configured steps)
4. **Trajectory Output** - Saved as `trajectory.dcd`

### Phase 2: Trajectory Analysis (Progress 70-85%)

Automated analysis using MDAnalysis:

| Metric | Description |
|--------|-------------|
| **RMSD** | Protein backbone stability over time |
| **RMSF** | Per-residue flexibility |
| **Radius of Gyration** | Protein compactness |
| **Ligand RMSD** | Ligand stability in binding site |
| **Binding Distance** | Minimum protein-ligand contact |

**Plots Generated:**
- `rmsd.png` - RMSD vs time
- `rmsf.png` - Flexibility per residue

### Phase 3: Interaction Profiling (Progress 85-95%)

Using PLIP to identify:
- Hydrogen bonds
- Hydrophobic contacts
- Pi-stacking interactions
- Salt bridges
- Water-mediated bridges

**Output:** `interactions.csv`

### Phase 4: Results (Progress 100%)

The worker returns:
```json
{
  "status": "completed",
  "analysis": {
    "rmsd_mean": 2.34,
    "rmsd_max": 3.87,
    "flexible_residues": 12,
    "hydrogen_bonds": 5,
    "hydrophobic_contacts": 8
  }
}
```

---

## Troubleshooting

### "No GPU found" Warning

**Solution:** Change runtime type to GPU
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Save

### "Task Rejected" or "Connection Refused"

**Cause:** Incorrect Redis URL

**Solution:**
1. Verify Redis URL in BioDockify dashboard
2. Ensure URL starts with `rediss://` (two 's')
3. Check for typos

### Worker Stops After 90 Minutes

**Cause:** Colab idle timeout

**Solution:** Keep the browser tab open and visible. Colab disconnects idle sessions after ~90 mins.

### "ModuleNotFoundError: openmm"

**Cause:** Conda installation failed

**Solution:** Re-run Cell 1 (installation cell). If it persists, restart runtime and try again.

---

## Performance Benchmarks

**On Google Colab T4 GPU:**

| System Size | Simulation Length | Time |
|-------------|-------------------|------|
| Small protein (~100 residues) | 10 ns | ~30-60 mins |
| Medium protein (~300 residues) | 10 ns | ~1-2 hours |
| Large protein (~500 residues) | 10 ns | ~3-5 hours |

**Comparison to CPU:**
- GPU is ~10-50x faster than CPU
- Free tier `t2.micro` would take days for the same job

---

## Limits & Quotas

### Google Colab Free Tier:

| Resource | Limit |
|----------|-------|
| Session Duration | 12 hours max |
| Idle Timeout | 90 minutes |
| GPU Usage | Dynamic quota (resets daily) |

**Best Practices:**
- Run simulations during off-peak hours
- Use "Save to Drive" for long trajectories
- Keep browser tab active

### Upstash Redis Free Tier:

| Resource | Limit |
|----------|-------|
| Commands/Day | 10,000 |
| Storage | 256 MB |

**Typical Usage:**
- 1 job = ~50 commands
- Free tier = ~200 jobs/day

---

## Advanced: Saving Results to Google Drive

Add this to Cell 4 (before the worker starts):

```python
from google.colab import drive
drive.mount('/content/drive')

# Modify work_dir in run_openmm_simulation():
work_dir = '/content/drive/My Drive/biodockify_results/' + job_id
```

This saves all trajectory files and analysis outputs to your Google Drive permanently.

---

## FAQ

**Q: Can I run multiple workers simultaneously?**  
A: Yes! Each user running the notebook = 1 worker. 100 users = 100 GPUs (all free).

**Q: Do I need to keep the notebook open?**  
A: Yes. Closing the tab stops the worker. Use Google Drive saving for long jobs.

**Q: Can I modify the simulation parameters?**  
A: Yes, but do it via the BioDockify web interface, not by editing the notebook.

**Q: What happens if my session times out mid-simulation?**  
A: The job fails. For simulations >2 hours, implement checkpointing (save progress every hour to resume later).

**Q: Is my data private?**  
A: Yes. Simulations run on YOUR Colab instance. BioDockify only stores job metadata and results you choose to upload.

---

## Next Steps

1. ✅ Test with a small protein (10 ps simulation)
2. 📊 Review analysis plots in Colab output
3. 🔬 Run production simulations (10-100 ns)
4. 📈 Download CSVs for publication figures

**Need Help?**  
Contact: support@biodockify.com  
Documentation: https://biodockify.com/docs
