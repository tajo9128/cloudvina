# BioDockify - Future Features Roadmap

## Saved for Later Implementation

---

## Completed Features ✅

| Feature | Version | Status |
|---------|---------|--------|
| **AI Result Explainer (DeepSeek)** | v3.0.0 | ✅ |
| **Advanced 3D Visualization (Zoom/Spin/Styles)** | v3.0.0 | ✅ |
| **Batch Docking (Virtual Screening)** | v3.0.0 | ✅ |
| **ODDT Integration (Scoring)** | v3.0.0 | ✅ |
| Blind Docking (Auto Cavity Detection) | v1.0 | ✅ |
| ADMET Prediction | v1.0 | ✅ |

---

## Upcoming Priority Features 🚧

### 1. Target Prediction
- Input molecule SMILES -> Predict protein targets
- Uses SwissTargetPrediction API
- *Status: Planned*

### 2. Community Target Library
- Curated list of 50+ ready-to-dock protein targets
- *Status: In Design*

### 3. User Accounts & History
- Save past jobs
- Re-analyze visualization
- *Status: In Progress*

---

## Zero-Cost Infrastructure Strategy 🚀
All planned features leverage free-tier services and open-source ecosystems to ensure sustainability without recurring costs.

| Component | Technology | Cost Strategy |
|-----------|------------|---------------|
| **Compute** | Render (Free Tier) | Optimized for memory-efficient docking |
| **Database** | Supabase (Free Tier) | 500MB free storage (sufficient for thousands of jobs) |
| **Storage** | Supabase Storage | File retention policies to manage free limits |
| **AI/ML** | DeepSeek / Pre-trained | Use API free tiers or pre-trained lightweight models |
| **External APIs** | SwissTargetPrediction | Free academic APIs |

### Future Scalability (Zero Cost)
- **Client-Side Processing:** Move visualization and simple calculations to the browser to save server resources.
- **Community Data:** Use public datasets (PDB, ZINC) instead of paid libraries.
