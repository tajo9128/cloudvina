# 🛠️ Tier 2 & 3 AI Implementation Plan: "The Chemical Bridge"
**Objective:** To upgrade the BioDockify AI from a standard classifier to a **3-Tier Hierarchical Model** capable of "Zero-Shot" transfer to plant extracts.

---

## 📅 Architecture Overview (Recap)
*   **Tier 1:** Polyglot Pre-training (Already Completed via MolFormer-XL foundation).
*   **Tier 2 (New):** **Contrastive Pharmacophore Alignment (CPA)** (The Bridge).
*   **Tier 3 (New):** **Disease-Specific Fine-Tuning** (The Application).

---

## 📋 Step-by-Step Implementation Strategy

### Phase A: Data Preparation (Day 1-2)
1.  **Construct "Positive Pairs" (Triplets):**
    *   **Goal:** Create pairs of (Synthetic, Natural) that "should" be close.
    *   **Script:** `generate_pharmacophore_pairs.py`.
    *   **Logic:**
        *   Take ChEMBL Active (e.g., Donepezil).
        *   Take COCONUT Active (e.g., Huperzine A).
        *   *Condition:* Similarity score > 0.4 OR Shared Target (AChE).
2.  **Construct "Negative Samples":**
    *   Select random COCONUT molecules with *different* pharmacophores.

### Phase B: Tier 2 - The Contrastive Bridge (Day 3-5)
1.  **Custom Loss Function:**
    *   Implement `ContrastiveTripletLoss` in PyTorch.
    *   Formula: `loss = max(0, dist(anchor, positive) - dist(anchor, negative) + margin)`.
2.  **Model Modification:**
    *   Add a **Projection Head** (Linear -> ReLU -> Linear) on top of MolFormer.
    *   *Why?* To project embeddings into the "Contrastive Space" without destroying the pre-trained weights.
3.  **Training Loop:**
    *   Input: Batch of Triplets (Anchor, Pos, Neg).
    *   Output: Embeddings.
    *   Optimization: Minimize Triplet Loss.

### Phase D: Rigorous Validation (The "PhD Standard")
1.  **k-Fold Cross-Validation:**
    *   Instead of a simple "Train/Test" split, implement **5-Fold Stratified CV**.
    *   *Why?* It proves your accuracy isn't just luck on a "good split".
2.  **Advanced Metrics:**
    *   Generate a full **Classification Report** (Precision, Recall, F1-Score).
    *   Plot a **Confusion Matrix** (True Positives vs False Positives).
3.  **External Validation:**
    *   Test the model on a *completely separate* dataset (e.g., BindingDB or a newly curated plant list) to prove "Generalizability".

---

## 💻 Tech Stack & Files to Create
1.  **`ai/dataset_generator.py`**: Logic for creating Synthetic-Natural Triplets.
2.  **`ai/models/bridge_model.py`**: Definition of the MolFormer with Projection Head.
3.  **`ai/trainers/tier2_trainer.py`**: The training loop with Triplet Loss.
4.  **`ai/trainers/tier3_trainer.py`**: The standard fine-tuning loop.

## 🧪 Validation Metrics
*   **Metric 1 (Tier 2):** **Alignment Score**. (Do Synthetic and Natural actives cluster together in t-SNE?).
*   **Metric 2 (Tier 3):** **AUC-ROC** + **F1-Score** (CV Average).
