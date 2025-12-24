# CNS Ensemble B - Complete Training Plan
## Multi-Encoder AI for CNS Drug Discovery

**Environment**: Google Colab Pro + NVIDIA L4 GPU (24 GB VRAM)  
**Timeline**: 6 weeks to production  
**Cost**: $13/month (Colab Pro only)

---

## Overview

CNS Ensemble B is a federated multi-encoder AI system for CNS-specific drug discovery, combining:
- **ChemBERTa**: SMILES transformer for 768-dim molecular embeddings
- **GNN (GAT)**: Graph attention network for structure-aware features (256-dim)
- **ProteinCNN**: 1D CNN for protein sequence encoding (128-dim)
- **DeepDTA**: Fused affinity prediction model (pKd output)

---

## Week 1-2: ChemBERTa Fine-Tuning

### Objectives
- Fine-tune ChemBERTa on CNS-specific molecular properties
- Train multi-task heads for solubility, pKa, LogP, hygroscopicity
- Generate 768-dim embeddings for downstream fusion

### Dataset Requirements
- **Size**: 5,000-10,000 compounds
- **Sources**: DrugBank, ChEMBL, PubChem, plant alkaloids
- **Labels**: Solubility class (4 classes), pKa, LogP, hygroscopicity

### Training Setup
```python
# Google Colab Notebook: 01_chemberta_cns_plants.ipynb
Model: DeepChem/ChemBERTa-77M-MTR
Batch Size: 64
Learning Rate: 2e-5
Epochs: 5-10
Mixed Precision: Enabled (FP16)
GPU Usage: ~18 GB VRAM
```

### Key Features
- Multi-task learning (4 properties)
- [CLS] token as molecular representation
- Automatic tokenization (max length: 128)

### Validation Metrics
- Solubility classification accuracy: >75%
- pKa regression RMSE: <1.5
- LogP regression RMSE: <0.8

---

## Week 3-4: GNN Training

### Objectives
- Train Graph Attention Network on molecular graphs
- Learn structure-aware features complementing ChemBERTa
- Output 256-dim graph embeddings

### Architecture
```python
# Google Colab Notebook: 02_gnn_cns_plants.ipynb
Layers: 2x GATConv (4 attention heads each)
Node Features: Atomic number, degree, charge, aromaticity, H-count
Hidden Dim: 128
Output Dim: 256
Pooling: Global mean pooling
```

### Training Configuration
- Optimizer: Adam (lr=1e-3)
- Epochs: 20
- Batch Size: 64
- Task: Solubility classification (same as ChemBERTa)

### Data Preparation
- Convert SMILES → PyTorch Geometric graphs
- Bidirectional edges for all bonds
- Self-loops for single-atom molecules

---

## Week 5: DeepDTA Integration

### Objectives
- Build fused DeepDTA model combining ChemBERTa + GNN + ProteinCNN
- Train on drug-target affinity pairs
- Predict pKd values for CNS targets

### Dataset Requirements
- **Size**: 1,000-3,000 pairs per target
- **Targets**: AChE, BACE1, GSK-3β, Tau
- **Sources**: ChEMBL, BindingDB
- **Labels**: pKd (−log₁₀(Kd in M))

### Model Architecture
```python
# Google Colab Notebook: 03_deepdta_cns_system.ipynb
Ligand Path:
  - ChemBERTa (768) + GNN (256) = 1024-dim
  - MLP → 256-dim

Protein Path:
  - ProteinCNN (1D Conv) → 128-dim
  - MLP → 256-dim

Fusion:
  - Concatenate → 512-dim
  - MLP → 1 (pKd output, 0-1 range via Sigmoid)
```

### Training Configuration
- Loss: MSE (normalized pKd)
- Optimizer: Adam (lr=1e-4)
- Epochs: 30-50
- Batch Size: 32

---

## Week 6: Production Deployment

### Objectives
- Save all trained models to Google Drive
- Create inference pipeline
- Deploy to AWS Lambda (serverless)

### Model Checkpoints
```
/content/drive/MyDrive/
├── chemberta_cns_plants_v1/
├── gnn_cns_plants_v1.pt
├── gnn_clf_head_v1.pt
├── protein_cnn_v1.pt
└── deepdta_fused_v1.pt
```

### Inference API
```python
def predict_affinity(smiles: str, target_seq: str) -> float:
    """
    Input: SMILES + protein sequence
    Output: pKd (affinity score)
    
    Pipeline:
    1. ChemBERTa embedding (768)
    2. GNN embedding (256)
    3. ProteinCNN encoding (128)
    4. DeepDTA fusion → pKd
    """
```

### AWS Deployment
- **Service**: Lambda (Python 3.11)
- **Memory**: 3 GB (models: ~800 MB total)
- **Timeout**: 30s
- **Cold Start**: ~5s
- **Warm Inference**: ~200ms

---

## Validation Benchmarks

### Blood-Brain Barrier (BBB) Classification
- **Dataset**: B3DB (public CNS dataset)
- **Metric**: ROC-AUC >0.85
- **Test Set**: 500 compounds

### Affinity Prediction (DeepDTA)
- **Dataset**: DUD-E CNS subset
- **Metric**: Pearson R >0.75, RMSE <1.2
- **Test Set**: 200 pairs per target

---

## Hardware Requirements

### Google Colab Pro L4
- **GPU**: NVIDIA L4 (24 GB VRAM)
- **RAM**: 51 GB
- **Disk**: 225 GB
- **Cost**: $13/month
- **Recommended Runtime**: 4-6 hours/day

### Local Development (Optional)
- **GPU**: RTX 3080 (10 GB minimum)
- **RAM**: 32 GB
- **Frameworks**: PyTorch 2.x, CUDA 12.x

---

## Expected Results

### ChemBERTa
- **Training Time**: 2-3 hours
- **Accuracy**: 75-80% (solubility)
- **Embedding Quality**: High semantic clustering

### GNN
- **Training Time**: 1-2 hours
- **Accuracy**: 70-75% (solubility)
- **Structure Sensitivity**: Superior to ChemBERTa on rigid molecules

### DeepDTA
- **Training Time**: 4-6 hours
- **Affinity RMSE**: <1.2 pKd units
- **Pearson R**: >0.75

---

## Next Steps After Training

1. **Integration**: Deploy to ai.biodockify.com subdomain
2. **API**: RESTful endpoint for screening requests
3. **UI**: Real-time prediction dashboard
4. **Monitoring**: MLflow tracking for continuous improvement
