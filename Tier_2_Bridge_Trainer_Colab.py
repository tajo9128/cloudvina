# BioDockify Tier 2: The "Chemical Bridge" Trainer (MolFormer Edition)
# --------------------------------------------------------------------------------
# PhD Publication 1: "Development of a Hierarchical Deep Learning Framework..."
# Tier 2: Semantic Bridge via Contrastive Pharmacophore Alignment (CPA)
# --------------------------------------------------------------------------------

# 1. INSTALLATION (Run this cell first in Colab)
# !pip install transformers accelerate datasets torch rdkit-pypi scikit-learn

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random

# --------------------------------------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------------------------------------
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct" # The "Polyglot" Base (Tier 1)
MAX_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 3
PROJECTION_DIM = 256  # Size of the "Bridge" embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------------------
# 3. REAL DATA GENERATOR (Triplet Creation via Tanimoto Similarity)
# --------------------------------------------------------------------------------
class PharmacophoreTripletDataset(Dataset):
    def __init__(self, tokenizer, synthetic_csv, natural_csv, num_samples=5000):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        
        print(f"Loading datasets from {synthetic_csv} and {natural_csv}...")
        # Expect columns: ['smiles', 'pIC50', 'target_id']
        # For Demo purposes, we fallback to mock data if files don't exist
        try:
            self.syn_df = pd.read_csv(synthetic_csv)
            self.nat_df = pd.read_csv(natural_csv)
            self.real_data = True
            print(">>> Real Data Loaded Successfully.")
        except Exception as e:
            print(f">>> CSV load failed ({e}), switching to MOCK MODE for functionality verification.")
            self.real_data = False
            self.syn_df = pd.DataFrame({'smiles': ["CN1CCC[C@H]1C2=CC(=CC=C2)OC"]*100})
            self.nat_df = pd.DataFrame({'smiles': ["COc1cc(C=C)ccc1O"]*100})

        # Pre-calculate Fingerprints for fast similarity search
        print("Pre-calculating fingerprints...")
        self.syn_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024) 
                        for s in self.syn_df['smiles'] if Chem.MolFromSmiles(s)]
        self.nat_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024) 
                        for s in self.nat_df['smiles'] if Chem.MolFromSmiles(s)]
        
        # Filter valid SMILES
        self.syn_smiles = [s for s in self.syn_df['smiles'] if Chem.MolFromSmiles(s)]
        self.nat_smiles = [s for s in self.nat_df['smiles'] if Chem.MolFromSmiles(s)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Select Anchor (Synthetic)
        anchor_idx = random.randint(0, len(self.syn_smiles)-1)
        anchor_smi = self.syn_smiles[anchor_idx]
        anchor_fp = self.syn_fps[anchor_idx]
        
        # 2. Find Positive (Natural with High Similarity > 0.4)
        # In production, use FAISS or pre-computed neighbor lists for speed.
        # Here we do a random search retry loop for simplicity.
        positive_smi = None
        for _ in range(50): # Try 50 times to find a match
            cand_idx = random.randint(0, len(self.nat_smiles)-1)
            cand_fp = self.nat_fps[cand_idx]
            sim = DataStructs.TanimotoSimilarity(anchor_fp, cand_fp)
            if sim > 0.4: # Threshold for "Pharmacophore Match"
                positive_smi = self.nat_smiles[cand_idx]
                break
        
        # Fallback if no high-sim match found (to keep training moving)
        if positive_smi is None:
            positive_smi = random.choice(self.nat_smiles)

        # 3. Find Negative (Natural with Low Similarity < 0.2)
        negative_smi = None
        for _ in range(50):
            cand_idx = random.randint(0, len(self.nat_smiles)-1)
            cand_fp = self.nat_fps[cand_idx]
            sim = DataStructs.TanimotoSimilarity(anchor_fp, cand_fp)
            if sim < 0.2: 
                negative_smi = self.nat_smiles[cand_idx]
                break
                
        if negative_smi is None:
            negative_smi = random.choice(self.nat_smiles)

        # 4. Tokenize
        cols = ['anchor', 'positive', 'negative']
        smiles_list = [anchor_smi, positive_smi, negative_smi]
        inputs = {}
        
        for name, smi in zip(cols, smiles_list):
            try:
                tokenized = self.tokenizer(
                    smi, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=MAX_LENGTH, 
                    return_tensors="pt"
                )
                inputs[name + '_input_ids'] = tokenized['input_ids'].squeeze(0)
                inputs[name + '_attention_mask'] = tokenized['attention_mask'].squeeze(0)
            except:
                # Handle tokenizer errors on weird SMILES
                tokenized = self.tokenizer("C", padding='max_length', max_length=MAX_LENGTH, return_tensors="pt")
                inputs[name + '_input_ids'] = tokenized['input_ids'].squeeze(0)
                inputs[name + '_attention_mask'] = tokenized['attention_mask'].squeeze(0)
            
        return inputs

# --------------------------------------------------------------------------------
# 4. THE CHEMICAL BRIDGE ARCHITECTURE (MolFormer + CPA Head)
# --------------------------------------------------------------------------------
class BioDockifyBridgeModel(nn.Module):
    def __init__(self, model_name, projection_dim=256):
        super().__init__()
        # Load Tier 1 (Pre-trained Polyglot)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Tier 2: The "Bridge" Projection Head
        # Projects both Synthetic and Natural vectors into a shared semantic space
        self.bridge_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        # 1. Get Base Embeddings (Tier 1)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token state or Mean Pooling
        cls_embedding = outputs.pooler_output 
        
        # 2. Project to Bridge Space (Tier 2)
        bridge_embedding = self.bridge_head(cls_embedding)
        return bridge_embedding

# --------------------------------------------------------------------------------
# 5. CONTRASTIVE TRIPLET LOSS
# --------------------------------------------------------------------------------
class ContrastiveTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.p_dist = nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        # Calculate Euclidean distances
        dist_pos = self.p_dist(anchor, positive)
        dist_neg = self.p_dist(anchor, negative)
        
        # Loss = max(0, dist_pos - dist_neg + margin)
        # We want Positive to be CLOSER to Anchor than Negative is (by a margin)
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return loss.mean()

# --------------------------------------------------------------------------------
# 6. TRAINING LOOP
# --------------------------------------------------------------------------------
def train_tier_2():
    print(">>> Loading Tokenizer (Polyglot)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print(">>> Initializing Chemical Bridge Model...")
    model = BioDockifyBridgeModel(MODEL_NAME, PROJECTION_DIM).to(device)
    
    print(">>> Generating Synthetic-Natural Triplets...")
    # UPDATE THESE PATHS TO YOUR REAL CSVs ON COLAB
    SYNTHETIC_PATH = "chembl_alzheimers.csv" 
    NATURAL_PATH = "coconut_natural_products.csv"
    
    train_dataset = PharmacophoreTripletDataset(tokenizer, SYNTHETIC_PATH, NATURAL_PATH, num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = ContrastiveTripletLoss(margin=1.0)
    
    model.train()
    print(">>> Starting Contrastive Training (Tier 2 Link)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move batch to device
            a_ids = batch['anchor_input_ids'].to(device)
            a_mask = batch['anchor_attention_mask'].to(device)
            p_ids = batch['positive_input_ids'].to(device)
            p_mask = batch['positive_attention_mask'].to(device)
            n_ids = batch['negative_input_ids'].to(device)
            n_mask = batch['negative_attention_mask'].to(device)
            
            # Forward Pass: Get Embeddings in Bridge Space
            anchor_emb = model(a_ids, a_mask)
            positive_emb = model(p_ids, p_mask)
            negative_emb = model(n_ids, n_mask)
            
            # Calculate Triplet Loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Average Triplet Loss: {avg_loss:.4f}")

    print(">>> Tier 2 Training Complete!")
    print(">>> The 'Chemical Bridge' is now constructed. The Model is ready for Tier 3 Fine-tuning.")
    
    # Save the "Bridged" Model
    # torch.save(model.state_dict(), "tier2_bridge_model.pth")
    return model

if __name__ == "__main__":
    # Uncomment to run training
    # trained_model = train_tier_2()
    pass
