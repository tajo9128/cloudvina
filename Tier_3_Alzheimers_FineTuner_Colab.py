# BioDockify Tier 3: The "Disease Specialist" (Alzheimer's Fine-Tuning)
# --------------------------------------------------------------------------------
# PhD Publication 1: "Development of a Hierarchical Deep Learning Framework..."
# Tier 3: Fine-tuning the "Bridged" Model on AChE/BACE1 Targets
# --------------------------------------------------------------------------------

# 1. INSTALLATION (Run this cell first in Colab)
# !pip install transformers accelerate datasets torch rdkit-pypi scikit-learn

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# --------------------------------------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------------------------------------
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
TIER_2_WEIGHTS = "tier2_bridge_model.pth" # Output from previous script
MAX_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-5 # Lower LR for fine-tuning
EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------------------
# 3. RE-DEFINE THE BRIDGE MODEL ARCHITECTURE (Must match Tier 2)
# --------------------------------------------------------------------------------
class BioDockifyBridgeModel(nn.Module):
    def __init__(self, model_name, projection_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.encoder.config.hidden_size
        self.bridge_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.pooler_output 
        bridge_embedding = self.bridge_head(cls_embedding)
        return bridge_embedding

# --------------------------------------------------------------------------------
# 4. DEFINE THE TIER 3 CLASSIFIER (The Specialist)
# --------------------------------------------------------------------------------
class BioDockifyAlzheimersSpecialist(nn.Module):
    def __init__(self, bridge_model, frozen=True):
        super().__init__()
        self.bridge_model = bridge_model
        
        # Freeze the Bridge (optional: unfreeze later)
        if frozen:
            for param in self.bridge_model.parameters():
                param.requires_grad = False
                
        # The Final Classification Head (Activity Prediction)
        # Input is the Dimension of the Bridge Projection (256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask):
        # get bridged embeddings
        embeddings = self.bridge_model(input_ids, attention_mask)
        # classify
        output = self.classifier(embeddings)
        return output

# --------------------------------------------------------------------------------
# 5. DATASET (ChEMBL Classification)
# --------------------------------------------------------------------------------
class AlzheimerDataset(Dataset):
    def __init__(self, tokenizer, data_path, split='train'):
        self.tokenizer = tokenizer
        
        # Load and Filter
        try:
            df = pd.read_csv(data_path)
            # Binary Label: Active if pIC50 > 7.0
            df['label'] = (df['pIC50'] >= 7.0).astype(int)
            
            # Simple Train/Val split
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
            
            self.data = train_df if split == 'train' else val_df
            self.smiles = self.data['smiles'].tolist()
            self.labels = self.data['label'].tolist()
            
        except Exception as e:
            print(f"Error loading data: {e}. Using Dummy Data.")
            self.smiles = ["CN1CCC[C@H]1C2=CC(=CC=C2)OC", "CCOc1ccccc1C(=O)N"] * 50
            self.labels = [1, 0] * 50

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = str(self.smiles[idx])
        label = float(self.labels[idx])
        
        tokenized = self.tokenizer(
            smi, 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_LENGTH, 
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# --------------------------------------------------------------------------------
# 6. TRAINING LOOP
# --------------------------------------------------------------------------------
def train_tier_3():
    print(">>> Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 1. Load the Pre-Trained Bridge
    print(f">>> Loading Tier 2 Bridge Weights from {TIER_2_WEIGHTS}...")
    bridge_model = BioDockifyBridgeModel(MODEL_NAME, projection_dim=256)
    
    try:
        bridge_model.load_state_dict(torch.load(TIER_2_WEIGHTS, map_location=device))
        print(">>> Bridge Weights Loaded Successfully!")
    except:
        print(">>> WARNING: Tier 2 weights not found. Using random weights (Just for demo).")
    
    # 2. Init Specialist
    model = BioDockifyAlzheimersSpecialist(bridge_model, frozen=True).to(device)
    
    # 3. Data
    DATA_PATH = "chembl_alzheimers.csv"
    train_dataset = AlzheimerDataset(tokenizer, DATA_PATH, split='train')
    val_dataset = AlzheimerDataset(tokenizer, DATA_PATH, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # 4. Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)
            
            preds = model(ids, mask)
            loss = criterion(preds, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].numpy()
                
                preds = model(ids, mask).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f} | Val AUC = {auc:.4f}")

    print(">>> Tier 3 Training Complete!")
    print(">>> Saving Final BioDockify Model...")
    torch.save(model.state_dict(), "biodockify_final_model.pth")

if __name__ == "__main__":
    train_tier_3()
