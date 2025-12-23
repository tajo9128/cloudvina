# BioDockify: Plant Ensemble Screener (Deep Learning Tier 3)
# Reads Plant Data from biodockify_global_plant_database.csv
# Uses Model: Tier 3 Ensemble (MolFormer + ChemBERTa) (91% Accuracy)

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from tqdm.auto import tqdm
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = "biodockify_ensemble_fold1.pth" # The model we just trained
MOLFORMER_NAME = "ibm/MoLFormer-XL-both-10pct"
CBERTA_NAME = "seyonec/ChemBERTa-zinc-base-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. ARCHITECTURE (Must match Training Script) ---
class BiModalEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.mol_enc = AutoModel.from_pretrained(MOLFORMER_NAME, trust_remote_code=True)
        self.cberta_enc = AutoModel.from_pretrained(CBERTA_NAME)
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        
    def forward(self, mf_ids, mf_mask, cb_ids, cb_mask):
        mf_out = self.mol_enc(mf_ids, mf_mask).pooler_output
        cb_out = self.cberta_enc(cb_ids, cb_mask).pooler_output
        return self.classifier(torch.cat((mf_out, cb_out), dim=1))

# --- 3. DATA LOADER (From MD Guide) ---
# --- 3. DATA LOADER (From MD Guide) ---
def load_plant_data():
    """
    Attempts to load plant data from sources.
    """
    print(">>> 📂 Loading Plant Data Sources...")
    
    # 0. Mount Drive (CRITICAL for Colab)
    try:
        from google.colab import drive
        if not os.path.isdir("/content/drive"):
            drive.mount('/content/drive')
    except: pass
    
    # 3.1 Check for Real Files
    # We prioritize the "Global Database" created by the Fetcher
    files = [
        "biodockify_global_plant_database.csv", 
        "/content/drive/MyDrive/biodockify_global_plant_database.csv",
        "imppat_phytochemicals.csv", 
        "/content/drive/MyDrive/BioDockify/biodockify_global_plant_database.csv"
    ]
    combined_smiles = []
    
    data_found = False
    for f in files:
        if os.path.exists(f):
            print(f"    ✅ Found Database: {f}")
            try:
                df = pd.read_csv(f)
                # Flexible Column Names
                for col in ['smiles', 'SMILES', 'Canonical_SMILES']:
                    if col in df.columns:
                        combined_smiles.extend(df[col].dropna().tolist())
                        data_found = True
                        break
            except Exception as e:
                print(f"    ⚠️ Error reading {f}: {e}")
            
            if data_found: break # Stop if we found the big database

    # 3.2 STRICT CHECK (No More Demo Mode)
    if not combined_smiles:
        print("\n❌ STOCK ERROR: No Plant Data Found!")
        print("    You must run 'BioDockify_Plant_Data_Fetcher.py' FIRST.")
        print("    That script downloads the thousands of SMILES you need.")
        print("    Please run the Fetcher, then come back here.")
        raise FileNotFoundError("Run the Fetcher Script first!")

    print(f"    Total Compounds to Screen: {len(combined_smiles)}")
    return combined_smiles

    print(f"    Total Compounds to Screen: {len(combined_smiles)}")
    return combined_smiles

# --- 4. SCREENING ENGINE ---
def screen_plants():
    print(f"\n>>> 🚀 Starting High-Performance Tier 3 Screening...")
    
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model '{MODEL_PATH}' not found! validation requires trained model.")
        return

    model = BiModalEnsemble().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("    ✅ Ensemble Model Loaded.")

    # Load Data
    smiles_list = load_plant_data()
    
    # Predict
    results = []
    mf_tok = AutoTokenizer.from_pretrained(MOLFORMER_NAME, trust_remote_code=True)
    cb_tok = AutoTokenizer.from_pretrained(CBERTA_NAME)
    
    print("    Running Inference Loop...")
    batch_size = 8 # Small batch for dual model
    
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            try:
                # Tokenize Stream 1
                mf_enc = mf_tok(batch, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
                # Tokenize Stream 2
                cb_enc = cb_tok(batch, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
                
                logits = model(
                    mf_enc['input_ids'].to(DEVICE), mf_enc['attention_mask'].to(DEVICE),
                    cb_enc['input_ids'].to(DEVICE), cb_enc['attention_mask'].to(DEVICE)
                )
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
                for s, p in zip(batch, probs):
                    results.append({'smiles': s, 'confidence': p})
            except Exception as e:
                pass # Skip bad SMILES
                
    # Rank
    df_res = pd.DataFrame(results).sort_values(by='confidence', ascending=False)
    top_10 = df_res.head(10)
    
    print("\n🏆 TOP 10 PLANT CANDIDATES (Tier 3 Ensemble):")
    print(top_10)
    
    # Save
    df_res.to_csv("biodockify_plant_screen_results.csv", index=False)
    print("\n✅ Full results saved to 'biodockify_plant_screen_results.csv'")

if __name__ == "__main__":
    screen_plants()
