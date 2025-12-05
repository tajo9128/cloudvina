# BioDockify - Future Features Roadmap

## Saved for Later Implementation

---

## 1. Virtual Screening (Batch Docking) - HIGH PRIORITY

**Status:** Not Started  
**Effort:** 5 days  
**Running Cost:** $3-5 per 100 ligands

### What It Does
- Upload 10-1000 ligands at once
- Dock all against one receptor
- Get ranked results table

### Implementation Notes
- Use AWS Batch array jobs
- Need SDF/CSV file parser
- Frontend: Multi-file upload + progress tracking

---

## 2. Target Prediction - MEDIUM PRIORITY

**Status:** Not Started  
**Effort:** 1 day  
**Running Cost:** FREE (external API)

### What It Does
- Input molecule SMILES
- Predict which proteins it might bind
- Uses SwissTargetPrediction API

### Implementation Notes
- Just API integration + frontend panel
- External dependency on Swiss server

---

## Completed Features ✅

| Feature | Date | Status |
|---------|------|--------|
| Blind Docking (Auto Cavity Detection) | Dec 2024 | ✅ |
| Multiple Pocket Results | Dec 2024 | ✅ |
| H-Bond Visualization | Dec 2024 | ✅ |
| ADMET Prediction + Drug-Likeness | Dec 2024 | ✅ |

---

## Cost Summary

| Usage Level | Monthly AWS Cost |
|-------------|------------------|
| Low (200 jobs) | ~$8 |
| Medium (1500 jobs) | ~$60 |
| High (10000 jobs) | ~$400 |

Free tier services: Supabase, Render, RDKit, 3Dmol.js, SwissADME
