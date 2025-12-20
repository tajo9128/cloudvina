# 🧪 Tier 2 Validation Sprint: BioDockify Multi-Target Ensemble
**Objective:** Fine-tune ChemBERTa-2 on ~7K Alzheimer's inhibitors + PhD Plant Extracts for BioDockify Phase 3.
**Timeline:** Dec 14, 2025 - Jan 10, 2026
**Compute:** Google Colab Pro (L4 GPU / A100)

---

## 📅 3-Week Ultra-Compute Plan (Colab Pro / L4 GPU)

### **Week 1: The "Intelligence" Sprint (Training)**
*Goal: Train the World's Best Open Source Alzheimer's Model.*
- [ ] **Data Fusion (Day 1-2):** Run `level3_rigorous_training.py` to aggregate 100K+ compounds from PubChem/BindingDB.
- [ ] **MolFormer Training (Day 3-5):** Run `MolFormer_Training_Script.py` on **L4 GPU**.
  - **Objective:** Fine-tune 1.1 Billion Parameters with **bfloat16** precision.
  - **Output:** A model that understands chemical structure better than any standard algorithm.
- [ ] **Ensemble (Day 6-7):** Train the Tier 2 Ensemble (ChemBERTa + XGBoost) to assist MolFormer prediction.

### **Week 2: The "Creation" Sprint (Generative AI)**
*Goal: Use the AI to invent NEW drugs.*
- [ ] **Generative Sampling (Day 8-10):** 
  - Use MolFormer as a **Generator** (Masked Language Modeling) to dream up new structures.
  - Generate 10,000 novel candidates seeded from your PhD plant compounds.
- [ ] **Toxicity Screening (Day 11-12):** 
  - Filter these 10k candidates using our `toxicity_engine.py`.
  - Discard anything with high hepatotoxicity risk.
- [ ] **Explainability (Day 13-14):** 
  - Run **SHAP Analysis** on the top 10 survivors.
  - Visualize *exactly* which atoms make them potent (great for thesis figures).

### **Week 3: The "Physics" Sprint (Deep Docking & MD)**
*Goal: Prove they work in 3D (The "Gold Standard").*
- [ ] **Deep Docking (Day 15-18):** 
  - Install **DiffDock** (SOTA Deep Learning Docker) on L4 GPU.
  - Dock the Top 5 AI-generated compounds into BACE1 (PDB: 1FKN).
  - Compare poses with standard Vina results.
- [ ] **Molecular Dynamics (Day 19-21):** 
  - Run short **10ns MD Simulations** (OpenMM) on the #1 Candidate.
  - Create a video of your molecule binding to the protein.
  - **Result:** definitive computational proof for publication.

---

## 🔬 Experiment Design
**Hypothesis:** ChemBERTa fine-tuned on multi-target AD dataset outperforms baseline by 10% RMSE, utilizing weighted loss for plant extracts.

**Dataset (7,012 SMILES):**
- **ChEMBL:** 3,500 AChE/BACE1/GSK-3β inhibitors.
- **PubChem:** 1,800 bioactives + 1,200 literature compounds.
- **PhD Data:** 8-12 plant SMIs (10x weighted).

**Metrics:**
- Primary: RMSE
- Secondary: R², MAE

**Colab Setup:**
- Runtime: L4 GPU + High RAM (24GB VRAM)
- Libraries: `transformers`, `deepchem`, `rdkit`, `shap`, `openmm`
