# 🌿 CNS System-Specific AI – Ensemble B (ChemBERTa + GNN + DeepDTA)
## Complete From-Scratch Setup on Google Colab Pro (L4 GPU)

**Date:** December 23, 2025  
**Goal:** Build a CNS system-specific AI model for plant + chemical data  
**Environment:** Google Colab Pro with **NVIDIA L4 GPU** (24 GB VRAM)  
**Timeline:** 6 weeks to production-ready system  

---

## TABLE OF CONTENTS

1. [Environment Setup](#1-environment-setup)
2. [Data Design](#2-data-design-for-ensemble-b)
3. [Stage 1: Fine-Tune ChemBERTa](#3-stage-1--fine-tune-chemberta)
4. [Stage 2: Train GNN](#4-stage-2--train-gnn-for-structure-aware-features)
5. [Stage 3: Build DeepDTA](#5-stage-3--cns-deepdta-with-fused-chemberta--gnn)
6. [System Integration](#6-cns-system-level-inference)
7. [Production Deployment](#7-how-to-use-this-in-practice)

---

## 1. Environment Setup

### 1.1 Colab Pro with L4 GPU

1. Open **Google Colab Pro** (colab.research.google.com).
2. Go to: `Runtime → Change runtime type → Hardware accelerator → GPU → Select L4`
3. Verify GPU availability (L4 has 24 GB VRAM, ideal for transformers + GNN).

### 1.2 Install Core Libraries

In the first notebook cell, install everything:

```bash
!pip install rdkit-pypi torch torchvision torchaudio \
    torch-geometric torch-scatter torch-sparse \
    deepchem transformers datasets \
    scikit-learn pandas numpy scipy networkx
```

### 1.3 Verify GPU

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Expected output: `NVIDIA L4` with ~24 GB VRAM.

### 1.4 Notebook Organization

Create three separate notebooks:

- **`01_chembarta_cns_plants.ipynb`** – ChemBERTa fine-tune
- **`02_gnn_cns_plants.ipynb`** – GNN fine-tune
- **`03_deepdta_cns_system.ipynb`** – Fused DeepDTA training
- **`04_cns_inference.ipynb`** – Production inference

Each notebook saves trained models to Google Drive for persistence.

---

## 2. Data Design for Ensemble B

### 2.1 Ligand (Compound) Data

Create CSV: `cns_plants_properties.csv`

| Column | Type | Example | Purpose |
|--------|------|---------|---------|
| `smiles` | str | `CC(=O)Oc1ccccc1C(=O)O` | Molecular structure |
| `plant_source` | str | `Evolvulus`, `Cordia`, `synthetic` | Origin tracking |
| `sol_class` | int | 0, 1, 2, 3 | Solubility class (I–IV BCS) |
| `logp` | float | 2.1 | Lipophilicity |
| `pka` | float | 4.5 | Ionization constant |
| `tpsa` | float | 66.0 | Topological Polar Surface Area |
| `hbd` | int | 2 | H-bond donors |
| `hba` | int | 4 | H-bond acceptors |
| `rot_bonds` | int | 3 | Rotatable bonds |
| `bbb_class` | int | 0 or 1 | Blood-Brain Barrier penetration |
| `tox_flag` | int | 0 or 1 | Toxicity indicator |

**Data sources:**

- Your curated phytochemicals (Evolvulus, Cordia, etc.)
- DrugBank (free download, CNS drugs with ADMET data)
- ChEMBL (5M+ compounds with property annotations)
- PubChem (100M+ with basic properties)

**Target size:** 5,000–10,000 unique compounds with complete property labels.

### 2.2 Drug–Target Affinity Data

For each CNS target (AChE, BACE1, GSK-3β, tau):

Create CSV: `cns_dta_pairs.csv`

| Column | Type | Example | Purpose |
|--------|------|---------|---------|
| `smiles` | str | `CC(=O)Oc1ccccc1C(=O)O` | Drug molecule |
| `target_id` | str | `ache`, `bace1`, `gsk3b`, `tau` | Target protein |
| `target_seq` | str | `MSLLFVNTKL...` | Amino acid sequence |
| `pKd` | float | 6.5 | −log₁₀(Kd in M) |

**Data sources:**

- **ChEMBL:** Download CNS-related assays (Ki, Kd, IC₅₀ values)
- **BindingDB:** Large affinity database with SMILES + protein sequences
- **UniProt:** Get canonical sequences for human targets
- Convert Ki/IC₅₀ to pKd using standard formulas

**Target size:** 1,000–3,000 pairs per target (minimum 500 for fine-tuning).

---

## 3. Stage 1: Fine-Tune ChemBERTa

ChemBERTa is a SMILES transformer pre-trained on millions of drug-like molecules.

### 3.1 Create Notebook: `01_chembarta_cns_plants.ipynb`

```python
# =============================================================================
# STAGE 1: FINE-TUNE CHEMBERTA ON CNS PLANT PROPERTIES
# =============================================================================

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import os

# Mount Google Drive for data persistence
from google.colab import drive
drive.mount('/content/drive')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Verify GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### 3.2 Load Data

```python
# Load compound data
df = pd.read_csv('/content/drive/MyDrive/cns_plants_properties.csv')
print(f"Loaded {len(df)} compounds")
print(df.head())

# For primary task, use solubility class
smiles_list = df['smiles'].tolist()
sol_class_labels = df['sol_class'].astype(int).tolist()

print(f"Solubility class distribution: {pd.Series(sol_class_labels).value_counts().to_dict()}")
```

### 3.3 Dataset & DataLoader

```python
class SmilesDataset(Dataset):
    """Dataset for SMILES strings and property labels"""
    
    def __init__(self, smiles, labels, tokenizer, max_length=128):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        label = self.labels[idx]
        
        # Tokenize SMILES
        enc = self.tokenizer(
            smiles,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Flatten single-batch dimension
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        
        return item

# Load pretrained ChemBERTa
model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4  # 4 solubility classes
).to(device)

# Create dataset & loader
dataset = SmilesDataset(smiles_list, sol_class_labels, tokenizer)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

print(f"Created DataLoader with {len(loader)} batches")
```

### 3.4 Fine-Tune with Mixed Precision

```python
# Setup optimizer & loss scaling
optimizer = AdamW(model.parameters(), lr=2e-5)
scaler = GradScaler()
num_epochs = 5

print("=" * 80)
print("FINE-TUNING CHEMBERTA ON CNS PLANT PROPERTIES")
print("=" * 80)

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    batch_count = 0
    
    for batch in loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        batch_count += 1
        
        if batch_count % 10 == 0:
            print(f"  Epoch {epoch+1}, Batch {batch_count}/{len(loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}")

print("\n✓ ChemBERTa fine-tuning completed!")
```

### 3.5 Save & Define Embedding Function

```python
# Save fine-tuned model
output_dir = '/content/drive/MyDrive/chembarta_cns_plants_v1'
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✓ Model saved to {output_dir}")

def get_chemberta_embedding(smiles: str, model=model, tokenizer=tokenizer, device=device):
    """
    Get 768-dim ChemBERTa embedding for a SMILES string.
    
    Args:
        smiles: SMILES string
        model: Fine-tuned ChemBERTa model
        tokenizer: ChemBERTa tokenizer
        device: torch device
    
    Returns:
        numpy array of shape (768,)
    """
    model.eval()
    
    enc = tokenizer(
        smiles,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        # Access the underlying BERT encoder
        outputs = model.bert(**{k: enc[k] for k in ['input_ids', 'attention_mask']})
    
    # Use [CLS] token (index 0) as molecular representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  # (768,)
    
    return cls_embedding.detach().cpu().numpy()

# Test embedding
test_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
test_emb = get_chemberta_embedding(test_smiles)
print(f"Test embedding shape: {test_emb.shape}")
print(f"Test embedding (first 10): {test_emb[:10]}")
```

---

## 4. Stage 2: Train GNN for Structure-Aware Features

GNNs provide explicit molecular graph understanding, complementing ChemBERTa's sequential semantics.

### 4.1 Create Notebook: `02_gnn_cns_plants.ipynb`

```python
# =============================================================================
# STAGE 2: TRAIN GNN FOR STRUCTURE-AWARE MOLECULAR FEATURES
# =============================================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import GATConv, global_mean_pool
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

torch.manual_seed(42)
np.random.seed(42)
```

### 4.2 Convert SMILES to PyTorch Geometric Graphs

```python
def smiles_to_pyg(smiles: str):
    """
    Convert SMILES string to PyTorch Geometric Data object.
    
    Nodes: atoms with atomic features
    Edges: bonds (bidirectional)
    
    Returns:
        torch_geometric.data.Data or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features: atomic number, degree, formal charge, aromaticity, total H
    x = []
    for atom in mol.GetAtoms():
        features = [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(int(atom.GetIsAromatic())),
            float(atom.GetTotalNumHs())
        ]
        x.append(features)
    
    x = torch.tensor(x, dtype=torch.float)
    
    # Edge indices: bonds (bidirectional)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    
    if len(edges) == 0:
        # Single atom, self-loop
        edges = [[0, 0]]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

# Test
test_mol = smiles_to_pyg("CC(=O)Oc1ccccc1C(=O)O")
print(f"Graph nodes: {test_mol.x.shape[0]}, edges: {test_mol.edge_index.shape[1]}")
```

### 4.3 Define GNN Encoder

```python
class MolGNN(nn.Module):
    """
    Graph Attention Network for molecular graphs.
    
    Architecture:
    - Input: Atomic features (5-dim)
    - Hidden: 128-dim with 4 attention heads
    - Output: 256-dim molecular embedding
    """
    
    def __init__(self, in_dim=5, hidden=128, out_dim=256):
        super().__init__()
        
        # Graph Attention layers
        self.gat1 = GATConv(in_dim, hidden, heads=4, dropout=0.2)
        self.gat2 = GATConv(hidden * 4, hidden, heads=4, dropout=0.2)
        
        # Linear projection to output dimension
        self.lin = nn.Linear(hidden * 4, out_dim)
        
    def forward(self, data):
        """
        Forward pass through GNN.
        
        Args:
            data: torch_geometric.data.Data or Batch with x, edge_index, batch
        
        Returns:
            Tensor of shape (batch_size, out_dim) = (batch_size, 256)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First attention layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        
        # Second attention layer
        x = self.gat2(x, edge_index)
        x = torch.relu(x)
        
        # Global mean pooling (aggregate all atoms → single graph vector)
        x = global_mean_pool(x, batch)
        
        # Linear projection
        x = self.lin(x)
        
        return x

# Instantiate model
gnn = MolGNN(in_dim=5, hidden=128, out_dim=256).to(device)
print(f"GNN model created with {sum(p.numel() for p in gnn.parameters())} parameters")
```

### 4.4 Train GNN on Property Labels

```python
# Load data
df = pd.read_csv('/content/drive/MyDrive/cns_plants_properties.csv')

# Build graphs & labels
graphs = []
for idx, row in df.iterrows():
    smiles = row['smiles']
    sol_class = int(row['sol_class'])
    
    g = smiles_to_pyg(smiles)
    if g is not None:
        g.y = torch.tensor([sol_class], dtype=torch.long)
        graphs.append(g)

print(f"Created {len(graphs)} valid molecular graphs")

# Create DataLoader
loader = GeoLoader(graphs, batch_size=64, shuffle=True)

# Define classification head
clf_head = nn.Linear(256, 4).to(device)  # 4 solubility classes

# Training setup
opt = torch.optim.Adam(list(gnn.parameters()) + list(clf_head.parameters()), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("=" * 80)
print("TRAINING GNN ON SOLUBILITY CLASSIFICATION")
print("=" * 80)

gnn.train()
clf_head.train()

for epoch in range(20):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        opt.zero_grad()
        
        # Forward
        emb = gnn(batch)  # (batch_size, 256)
        logits = clf_head(emb)  # (batch_size, 4)
        loss = loss_fn(logits, batch.y)
        
        # Backward
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == batch.y).sum().item()
        total += batch.y.size(0)
    
    acc = 100 * correct / total
    print(f"Epoch {epoch+1:2d}: Loss = {total_loss/len(loader):.4f}, Accuracy = {acc:.2f}%")

print("\n✓ GNN training completed!")
```

### 4.5 Save & Define GNN Embedding Function

```python
# Save GNN
torch.save(gnn.state_dict(), '/content/drive/MyDrive/gnn_cns_plants_v1.pt')
torch.save(clf_head.state_dict(), '/content/drive/MyDrive/gnn_clf_head_v1.pt')
print("✓ GNN weights saved")

def get_gnn_embedding(smiles: str, gnn=gnn, device=device):
    """
    Get 256-dim GNN embedding for a SMILES string.
    
    Args:
        smiles: SMILES string
        gnn: Trained GNN model
        device: torch device
    
    Returns:
        numpy array of shape (256,)
    """
    g = smiles_to_pyg(smiles)
    if g is None:
        return None
    
    batch = Batch.from_data_list([g]).to(device)
    
    gnn.eval()
    with torch.no_grad():
        emb = gnn(batch).squeeze(0)  # (256,)
    
    return emb.detach().cpu().numpy()

# Test
test_emb = get_gnn_embedding("CC(=O)Oc1ccccc1C(=O)O")
print(f"GNN test embedding shape: {test_emb.shape}")
```

---

## 5. Stage 3: CNS DeepDTA with Fused ChemBERTa + GNN

DeepDTA predicts binding affinities using both ligand and target protein information.

### 5.1 Create Notebook: `03_deepdta_cns_system.ipynb`

```python
# =============================================================================
# STAGE 3: DEEPDTA WITH FUSED CHEMBERTA + GNN ENCODERS
# =============================================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from google.colab import drive
drive.mount('/content/drive')

torch.manual_seed(42)
np.random.seed(42)
```

### 5.2 Protein Sequence Encoder (CNN)

```python
# Amino acid vocabulary
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}
aa_to_idx[''] = 0  # padding

def encode_sequence(seq: str, max_len: int = 1000):
    """
    Encode amino acid sequence to integer indices.
    
    Args:
        seq: Protein sequence (FASTA string)
        max_len: Maximum sequence length (pad/truncate)
    
    Returns:
        List of integers (length = max_len)
    """
    ids = [aa_to_idx.get(aa, 0) for aa in seq[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids[:max_len]

class ProteinCNN(nn.Module):
    """
    1D CNN for protein sequences.
    
    Inspired by DeepDTA architecture:
    - Embedding: 32-dim per amino acid
    - Conv1d: 64 → 128 filters with max pooling
    - Output: 128-dim embedding per protein
    """
    
    def __init__(self, seq_len=1000, emb_dim=128):
        super().__init__()
        
        vocab_size = len(AA_VOCAB) + 1  # +1 for padding
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, 32, padding_idx=0)
        
        # 1D Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        
        # Fully connected layers
        # After two 3x pooling: seq_len // 9
        conv_output_len = seq_len // 9
        self.fc = nn.Linear(128 * conv_output_len, emb_dim)
    
    def forward(self, seq_ids):
        """
        Args:
            seq_ids: LongTensor of shape (batch_size, seq_len)
        
        Returns:
            Tensor of shape (batch_size, emb_dim)
        """
        # Embed: (B, L) → (B, L, 32) → (B, 32, L)
        x = self.embed(seq_ids).transpose(1, 2)
        
        # Conv layers
        x = self.conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc(x)
        
        return x

# Test
prot_cnn = ProteinCNN(seq_len=1000, emb_dim=128).to(device)
test_seq = encode_sequence("MSLLFVNTKL" * 100)
test_ids = torch.tensor([test_seq]).to(device)
test_out = prot_cnn(test_ids)
print(f"ProteinCNN output shape: {test_out.shape}")  # Should be (1, 128)
```

### 5.3 Fused DeepDTA Module

```python
class FusedDeepDTA(nn.Module):
    """
    DeepDTA-style model with fused ChemBERTa + GNN ligand encoders.
    
    Architecture:
    - Ligand path: 1024-dim (ChemBERTa 768 + GNN 256) → 256-dim
    - Protein path: 128-dim protein CNN output → 256-dim
    - Interaction: Concatenate → MLP → pKd (0-1 range)
    """
    
    def __init__(self, compound_dim=1024, protein_dim=128):
        super().__init__()
        
        # Ligand processing
        self.compound_fc = nn.Sequential(
            nn.Linear(compound_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Protein processing
        self.protein_fc = nn.Sequential(
            nn.Linear(protein_dim, 256),
            nn.ReLU()
        )
        
        # Interaction/affinity prediction
        self.interaction = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output: 0-1 (normalized pKd)
        )
    
    def forward(self, compound_feat, prot_feat):
        """
        Args:
            compound_feat: (batch_size, 1024) = ChemBERTa(768) + GNN(256)
            prot_feat: (batch_size, 128) = ProteinCNN output
        
        Returns:
            Tensor of shape (batch_size,) = predicted pKd (0-1 range)
        """
        # Process each modality
        c = self.compound_fc(compound_feat)  # (batch, 256)
        p = self.protein_fc(prot_feat)       # (batch, 256)
        
        # Concatenate
        x = torch.cat([c, p], dim=1)  # (batch, 512)
        
        # Predict affinity
        affinity = self.interaction(x).squeeze(1)  # (batch,)
        
        return affinity

# Test
dta_model = FusedDeepDTA(compound_dim=1024, protein_dim=128).to(device)
test_compound = torch.randn(2, 1024).to(device)
test_protein = torch.randn(2, 128).to(device)
test_affinity = dta_model(test_compound, test_protein)
print(f"DeepDTA output shape: {test_affinity.shape}")  # Should be (2,)
```

### 5.4 DTA Dataset Class

```python
class DtaDataset(Dataset):
    """
    Dataset for drug-target affinity pairs.
    
    Combines pre-computed ChemBERTa + GNN embeddings with protein sequences and labels.
    """
    
    def __init__(self, df, get_chemberta_fn, get_gnn_fn, encode_seq_fn):
        """
        Args:
            df: DataFrame with columns [smiles, target_seq, pKd]
            get_chemberta_fn: Function to get ChemBERTa embedding
            get_gnn_fn: Function to get GNN embedding
            encode_seq_fn: Function to encode protein sequence
        """
        self.df = df.reset_index(drop=True)
        self.get_chemberta = get_chemberta_fn
        self.get_gnn = get_gnn_fn
        self.encode_seq = encode_seq_fn
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['smiles']
        target_seq = row['target_seq']
        pkd = float(row['pKd'])
        
        # Get ligand embeddings
        c_emb = self.get_chemberta(smiles)   # (768,) numpy
        g_emb = self.get_gnn(smiles)         # (256,) numpy
        
        if g_emb is None:
            g_emb = np.zeros(256)
        
        compound_feat = np.concatenate([c_emb, g_emb])  # (1024,)
        compound_tensor = torch.from_numpy(compound_feat).float()
        
        # Encode protein sequence
        prot_ids = torch.tensor(self.encode_seq(target_seq), dtype=torch.long)
        
        # Target affinity (normalize to 0-1)
        pkd_tensor = torch.tensor(pkd / 15.0, dtype=torch.float32)  # Assume max pKd ~15
        
        return compound_tensor, prot_ids, pkd_tensor

# Example usage (after loading data):
# dta_df = pd.read_csv('/content/drive/MyDrive/cns_dta_pairs.csv')
# dta_dataset = DtaDataset(dta_df, get_chemberta_embedding, get_gnn_embedding, encode_sequence)
```

### 5.5 Training Loop

```python
# Load DTA data
dta_df = pd.read_csv('/content/drive/MyDrive/cns_dta_pairs.csv')
print(f"Loaded {len(dta_df)} drug-target pairs")

# Create dataset
dta_dataset = DtaDataset(
    dta_df,
    get_chemberta_embedding,
    get_gnn_embedding,
    encode_sequence
)

# Create DataLoader
dta_loader = DataLoader(dta_dataset, batch_size=128, shuffle=True)

# Initialize models
prot_encoder = ProteinCNN(seq_len=1000, emb_dim=128).to(device)
dta_model = FusedDeepDTA(compound_dim=1024, protein_dim=128).to(device)

# Setup training
params = list(prot_encoder.parameters()) + list(dta_model.parameters())
optimizer = Adam(params, lr=1e-3)
loss_fn = nn.MSELoss()

print("=" * 80)
print("TRAINING DEEPDTA ON CNS DRUG-TARGET AFFINITY")
print("=" * 80)

num_epochs = 30

for epoch in range(num_epochs):
    prot_encoder.train()
    dta_model.train()
    
    total_loss = 0
    
    for compound_feat, prot_ids, y_pkd in dta_loader:
        compound_feat = compound_feat.to(device)
        prot_ids = prot_ids.to(device)
        y_pkd = y_pkd.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        prot_emb = prot_encoder(prot_ids)  # (batch, 128)
        pred_affinity = dta_model(compound_feat, prot_emb)  # (batch,)
        
        loss = loss_fn(pred_affinity, y_pkd)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dta_loader)
    print(f"Epoch {epoch+1:2d}/{num_epochs}: MSE Loss = {avg_loss:.6f}")

print("\n✓ DeepDTA training completed!")
```

### 5.6 Save Models

```python
# Save protein encoder
torch.save(prot_encoder.state_dict(), '/content/drive/MyDrive/prot_encoder_cns_v1.pt')

# Save DeepDTA model
torch.save(dta_model.state_dict(), '/content/drive/MyDrive/dta_cns_v1.pt')

print("✓ Models saved to Google Drive")
```

---

## 6. CNS System-Level Inference

### 6.1 Create Notebook: `04_cns_inference.ipynb`

```python
# =============================================================================
# STAGE 4: PRODUCTION INFERENCE - CNS SYSTEM MODEL
# =============================================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from google.colab import drive
drive.mount('/content/drive')

# Load all trained models
model_dir = '/content/drive/MyDrive'

# Load ChemBERTa
model_name = "DeepChem/ChemBERTa-77M-MTR"
chemberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
chemberta = AutoModel.from_pretrained(model_name).to(device)
chemberta.load_state_dict(torch.load(f'{model_dir}/chemberta_cns_plants_v1/pytorch_model.bin', 
                                      map_location=device))

# Load GNN
# (assume MolGNN class is defined in this notebook or imported)
gnn = MolGNN(in_dim=5, hidden=128, out_dim=256).to(device)
gnn.load_state_dict(torch.load(f'{model_dir}/gnn_cns_plants_v1.pt', map_location=device))

# Load protein encoder
prot_encoder = ProteinCNN(seq_len=1000, emb_dim=128).to(device)
prot_encoder.load_state_dict(torch.load(f'{model_dir}/prot_encoder_cns_v1.pt', map_location=device))

# Load DeepDTA
dta_model = FusedDeepDTA(compound_dim=1024, protein_dim=128).to(device)
dta_model.load_state_dict(torch.load(f'{model_dir}/dta_cns_v1.pt', map_location=device))

print("✓ All models loaded successfully")
```

### 6.2 CNS Target Dictionary

```python
# Define CNS targets with their sequences (from UniProt or ChEMBL)
cns_targets = {
    "ache": {
        "full_name": "Acetylcholinesterase",
        "seq": "MSLLFVNTKL...",  # Truncated for brevity
        "encoded": None  # Will be cached
    },
    "bace1": {
        "full_name": "Beta-secretase 1",
        "seq": "MKFLKFSLLT...",
        "encoded": None
    },
    "gsk3b": {
        "full_name": "Glycogen synthase kinase-3 beta",
        "seq": "MAPGFTYDPV...",
        "encoded": None
    },
    "tau": {
        "full_name": "Tau protein",
        "seq": "MAEGEITTFT...",
        "encoded": None
    }
}

# Pre-encode sequences
for tgt_id, tgt_info in cns_targets.items():
    tgt_info['encoded'] = torch.tensor(encode_sequence(tgt_info['seq']), dtype=torch.long).to(device)
    print(f"Encoded {tgt_id}: {tgt_info['seq'][:30]}...")
```

### 6.3 Production Inference Function

```python
def cns_system_predict(smiles: str, verbose=True):
    """
    Comprehensive CNS prediction for a single phytochemical.
    
    Args:
        smiles: SMILES string of compound
        verbose: Print intermediate results
    
    Returns:
        Dictionary with:
        - per_target_pKd: Dict[target_id → pKd_pred]
        - cns_system_score: Aggregated score (0-1)
        - binding_summary: Human-readable summary
    """
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"CNS PREDICTION FOR: {smiles}")
        print(f"{'='*80}")
    
    # Step 1: Get ligand embeddings
    chemberta.eval()
    with torch.no_grad():
        c_enc = chemberta_tokenizer(
            smiles,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        c_out = chemberta(**{k: c_enc[k] for k in ['input_ids', 'attention_mask']})
        c_emb = c_out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # (768,)
    
    # Step 2: Get GNN embedding
    gnn.eval()
    g = smiles_to_pyg(smiles)
    if g is None:
        return None
    
    with torch.no_grad():
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([g]).to(device)
        g_emb = gnn(batch).squeeze(0).cpu().numpy()  # (256,)
    
    # Fuse embeddings
    compound_feat = np.concatenate([c_emb, g_emb])  # (1024,)
    compound_tensor = torch.from_numpy(compound_feat).float().unsqueeze(0).to(device)
    
    if verbose:
        print(f"Ligand embedding: ChemBERTa {c_emb.shape} + GNN {g_emb.shape}")
    
    # Step 3: Predict per-target affinity
    results = {}
    prot_encoder.eval()
    dta_model.eval()
    
    with torch.no_grad():
        for tgt_id, tgt_info in cns_targets.items():
            prot_ids = tgt_info['encoded'].unsqueeze(0)
            p_emb = prot_encoder(prot_ids)
            
            pkd_normalized = dta_model(compound_feat, p_emb).item()
            pkd_actual = pkd_normalized * 15.0  # Denormalize (assuming max pKd ~15)
            
            results[tgt_id] = {
                'pKd': pkd_actual,
                'pKd_normalized': pkd_normalized,
                'target_name': tgt_info['full_name'],
                'binding_strength': 'strong' if pkd_normalized > 0.6 else 'moderate' if pkd_normalized > 0.4 else 'weak'
            }
            
            if verbose:
                print(f"  {tgt_id:8s} ({tgt_info['full_name']:30s}): pKd={pkd_actual:.2f}, Binding={results[tgt_id]['binding_strength']}")
    
    # Step 4: Aggregate to system-level score
    system_score = np.mean([r['pKd_normalized'] for r in results.values()])
    
    # Step 5: Generate summary
    strong_targets = [k for k, v in results.items() if v['binding_strength'] == 'strong']
    
    summary = {
        'smiles': smiles,
        'per_target': results,
        'cns_system_score': system_score,
        'num_strong_binders': len(strong_targets),
        'strong_targets': strong_targets,
        'recommendation': 'Promising CNS candidate' if len(strong_targets) >= 2 else 'Moderate CNS potential' if len(strong_targets) == 1 else 'Weak CNS potential'
    }
    
    if verbose:
        print(f"\nCNS System Score: {system_score:.3f}")
        print(f"Strong binders ({len(strong_targets)}): {', '.join(strong_targets) if strong_targets else 'None'}")
        print(f"Recommendation: {summary['recommendation']}")
    
    return summary

# Test on example phytochemical
example_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin for testing
result = cns_system_predict(example_smiles, verbose=True)
```

### 6.4 Batch Processing

```python
def cns_batch_predict(smiles_list, plant_sources=None, top_n=20):
    """
    Predict CNS potential for multiple compounds and rank them.
    
    Args:
        smiles_list: List of SMILES strings
        plant_sources: Optional list of plant names (for tracking origin)
        top_n: Return top N ranked compounds
    
    Returns:
        DataFrame with ranked results
    """
    
    results_list = []
    
    for idx, smiles in enumerate(smiles_list):
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(smiles_list)}...")
        
        result = cns_system_predict(smiles, verbose=False)
        
        if result is not None:
            row = {
                'smiles': smiles,
                'plant_source': plant_sources[idx] if plant_sources else 'unknown',
                'cns_system_score': result['cns_system_score'],
                'num_strong_targets': result['num_strong_targets'],
                'strong_targets': ','.join(result['strong_targets']),
                'recommendation': result['recommendation']
            }
            
            # Add individual target scores
            for tgt_id, tgt_result in result['per_target'].items():
                row[f'{tgt_id}_pKd'] = tgt_result['pKd']
                row[f'{tgt_id}_binding'] = tgt_result['binding_strength']
            
            results_list.append(row)
    
    # Create DataFrame and sort by system score
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('cns_system_score', ascending=False)
    
    # Save results
    results_df.to_csv(f'{model_dir}/cns_predictions_results.csv', index=False)
    print(f"\n✓ Predictions saved to cns_predictions_results.csv")
    
    return results_df.head(top_n)

# Example: Batch predict your phytochemical library
# plant_df = pd.read_csv('your_plant_library.csv')
# top_cns = cns_batch_predict(plant_df['smiles'].tolist(), plant_df['source'].tolist())
# print(top_cns)
```

---

## 7. How to Use This in Practice

### 7.1 Complete Workflow

**Week 1–2: Data Preparation**
- Gather 5,000+ plant compounds with ADMET properties.
- Collect 1,000–3,000 drug–target affinity pairs per CNS target.
- Upload to Google Drive.

**Week 3–4: Model Training**
- Run `01_chembarta_cns_plants.ipynb` (2–3 hours).
- Run `02_gnn_cns_plants.ipynb` (4–5 hours).
- Run `03_deepdta_cns_system.ipynb` (6–8 hours).

**Week 5–6: Validation & Deployment**
- Run `04_cns_inference.ipynb` on your library.
- Validate predictions against known CNS drugs.
- Deploy on production server or Streamlit.

### 7.2 Extending to Other Systems

For CVS (cardiovascular), endocrine, etc., reuse ChemBERTa + GNN:

```python
# Simply change targets and retrain DeepDTA
cvs_targets = {
    "ace": {"seq": "...", "encoded": None},  # ACE inhibitor target
    "beta1": {"seq": "...", "encoded": None},  # Beta-1 receptor
    # ... more CVS targets
}

# Re-run `03_deepdta_cns_system.ipynb` with CVS target data
# → Results in CVS System Model v1
```

### 7.3 Integration with BioDockify Platform

```python
# Pseudo-code for integrating into your app
from cns_system_model import cns_system_predict

@app.route('/predict/cns', methods=['POST'])
def cns_prediction():
    smiles = request.json['smiles']
    result = cns_system_predict(smiles, verbose=False)
    return jsonify(result)
```

---

## 8. Troubleshooting & FAQs

### Q: Training is too slow
**A:** Reduce batch size (64 → 32) or use gradient accumulation. On L4 with 24GB VRAM, should handle batch 128.

### Q: Out of memory during ChemBERTa fine-tuning
**A:** Reduce sequence length from 128 to 64, or batch size from 64 to 32.

### Q: GNN embedding is None
**A:** Invalid SMILES string. Add validation: `if g is not None: ...`

### Q: DeepDTA predictions all similar
**A:** Model may be underfitting. Increase epochs from 30 to 50 or add more training data.

### Q: How do I validate accuracy?
**A:** Hold out 10–20% of each dataset as test set. Compute AUROC, RMSE, or Spearman correlation.

---

## 9. Production Deployment Checklist

- [ ] All three models trained and saved to Google Drive
- [ ] Inference notebook (`04_cns_inference.ipynb`) tested
- [ ] Batch prediction works on 100+ compounds
- [ ] Results validated against known CNS drugs
- [ ] API endpoint created for integration
- [ ] Documentation written
- [ ] Ensemble extended to other systems (CVS, endocrine)

---

## 10. References & Resources

- **ChemBERTa:** Chithrananda et al. (2020) – "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction"
- **DeepDTA:** Lee et al. (2019) – "DeepDTA: Deep Drug-Target Binding Affinity Prediction"
- **GraphDTA:** Nguyen et al. (2021) – "Graph Neural Networks for Drug-Target Binding Affinity Prediction"
- **PyTorch Geometric:** Fey & Lenssen (2019) – graph neural network library
- **UniProt:** Protein sequences and metadata
- **ChEMBL:** Drug and target affinity data

---

**Status:** ✅ **COMPLETE ENSEMBLE B SYSTEM READY FOR IMPLEMENTATION**

🚀 **Start with notebook `01_chembarta_cns_plants.ipynb` on Colab L4 today!**
