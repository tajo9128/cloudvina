# 🧬 BioDockify: Fine-Tune MolFormer-XL for Alzheimer's
# Compatible with Google Colab Free Tier (T4 GPU) and L4 GPU

import os
import sys
import shutil
import time
import logging
import json
import hashlib
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix)

# ==========================================
# PART 1: INSTALLATION & SETUP
# ==========================================
def install_dependencies():
    print("=" * 60)
    print("📦 INSTALLING DEPENDENCIES...")
    print("=" * 60)
    os.system('pip install -q transformers==4.36.2')
    os.system('pip install -q torch==2.1.0')
    os.system('pip install -q datasets')
    os.system('pip install -q accelerate')
    os.system('pip install -q pandas scikit-learn')
    os.system('pip install -q chembl_webresource_client --upgrade')
    print("✅ Core dependencies installed")

def check_system():
    print("\n" + "=" * 60)
    print("🖥️  SYSTEM SPECIFICATIONS")
    print("=" * 60)
    
    # GPU Check
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("❌ NO GPU DETECTED - Runtime will be slow")
        print("   On Colab: Runtime → Change Runtime Type → GPU (T4)")

    # RAM Check
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f"✅ System RAM: {ram_gb:.2f} GB")
    
    # Disk Check
    disk_usage = shutil.disk_usage('/')
    print(f"✅ Disk Space: {disk_usage.free / 1e9:.2f} GB free")

def setup_project():
    print("\n" + "=" * 60)
    print("📂 SETTING UP DIRECTORIES")
    print("=" * 60)
    project_dirs = [
        './biodockify_project',
        './biodockify_project/data',
        './biodockify_project/models',
        './biodockify_project/logs',
        './biodockify_project/results'
    ]
    
    for dir_path in project_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    os.chdir('./biodockify_project')
    print(f"✅ Working directory: {os.getcwd()}")

# ==========================================
# PART 2: DATA PREPARATION
# ==========================================
def fetch_data():
    # FORCE INSTALL if missing
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        print("⚠️  Library missing. Installing chembl_webresource_client...")
        os.system('pip install -q chembl_webresource_client')
        from chembl_webresource_client.new_client import new_client
    
    print("\n" + "=" * 60)
    print("📊 FETCHING ALZHEIMER'S DATA")
    print("=" * 60)
    
    targets = {
        'CHEMBL220': 'Acetylcholinesterase (AChE)',
        'CHEMBL2366': 'Beta-secretase 1 (BACE1)',
        'CHEMBL2366834': 'Tau protein'
    }
    
    all_data = []
    activity = new_client.activity
    
    for target_id, target_name in targets.items():
        print(f"\n🔍 Fetching {target_name} ({target_id})...")
        try:
            activities = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
            count = 0
            for res in activities:
                if res.get('standard_value') and res.get('canonical_smiles'):
                    try:
                        ic50 = float(res['standard_value'])
                        label = 1 if ic50 < 1000 else 0 # Active < 1000 nM
                        all_data.append({
                            'smiles': res['canonical_smiles'],
                            'ic50_value': ic50,
                            'target': target_name,
                            'active': label
                        })
                        count += 1
                        if count >= 800: break
                    except: continue
            print(f"   ✅ Retrieved {count} compounds")
        except Exception as e:
            print(f"   ⚠️ Could not fetch from ChEMBL: {e}")
            
    if not all_data:
        print("⚠️  ChEMBL API failed/empty. Generating synthetic fallback data...")
        return generate_fallback_data()
        
    df = pd.DataFrame(all_data)
    df_clean = df.drop_duplicates(subset=['smiles'])
    df_clean.to_csv('./data/alzheimers_compounds.csv', index=False)
    print(f"✅ Saved {len(df_clean)} unique compounds to ./data/alzheimers_compounds.csv")
    return df_clean

def generate_fallback_data():
    print("⚠️  Generating synthetic fallback data...")
    fallback_data = {
        'smiles': [
            'CN1C(=O)CC(c2c1cn[nH]2)(c1ccc(Cl)cc1)C(F)(F)F',
            'CC(=O)Oc1ccccc1C(=O)O',
            'COc1ccc2nc(sc2c1)S(=O)(=O)N', 
            'c1cc(c(cc1C(=O)O)O)O',
            'CC(C)Cc1ccc(cc1)C(C)C(=O)O'
        ] * 400,
        'ic50_value': np.random.uniform(100, 10000, 2000),
        'target': np.random.choice(['AChE', 'BACE1', 'Tau'], 2000),
        'active': np.random.choice([0, 1], 2000, p=[0.3, 0.7])
    }
    df = pd.DataFrame(fallback_data)
    df.to_csv('./data/alzheimers_compounds.csv', index=False)
    return df

class AlzheimersMolDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.smiles = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# PART 3: TRAINING
# ==========================================
def train_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    
    print("\n" + "=" * 60)
    print("⚙️  LOADING MODEL & TOKENIZER")
    print("=" * 60)
    
    model_name = "ibm/molformer-xl-both-10pct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, trust_remote_code=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load Data
    df = pd.read_csv('./data/alzheimers_compounds.csv')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['active'])
    
    train_dataset = AlzheimersMolDataset(train_df['smiles'].values, train_df['active'].values, tokenizer)
    val_dataset = AlzheimersMolDataset(val_df['smiles'].values, val_df['active'].values, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5) # Slightly higher LR for larger batch
    
    # Enable L4 optimizations (TF32/BF16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("🚀 Optimizing for L4 GPU (BF16 + TF32 enabled, Batch Size=32)...")
    
    num_epochs = 3
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)
    
    scaler = torch.cuda.amp.GradScaler() # Optional for bf16 but good for stability if mixed with fp16 layers
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # L4 supports BFloat16 natively
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
            
            # Backward pass with scaler (though bf16 rarely needs scaling, it's safer)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            
            if (batch_idx+1) % 10 == 0:
                print(f"   Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f"   ✅ Validation Accuracy: {val_acc:.4f}")
        
        # Save Checkpoint
        model.save_pretrained(f'./models/checkpoint_epoch{epoch+1}')

    print("\n✅ Training Complete. Saving best model...")
    model.save_pretrained('./models/best_model')
    return model, tokenizer, device, val_loader

# ==========================================
# PART 4: EVALUATION
# ==========================================
def evaluate_model(model, tokenizer, device, val_loader):
    print("\n" + "=" * 60)
    print("🧪 EVALUATION")
    print("=" * 60)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"📊 Accuracy: {acc:.4f}")
    print(f"📊 F1 Score: {f1:.4f}")
    
    # Test on Donepezil
    donepezil = "CN1C(=O)CC(c2c1cn[nH]2)(c1ccc(Cl)cc1)C(F)(F)F"
    inputs = tokenizer(donepezil, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        prob = torch.softmax(logits, dim=1)[0][1].item()
    
    print(f"\n🧬 Sample Prediction (Donepezil):")
    print(f"   Probability of Activity: {prob:.4f}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    install_dependencies()
    check_system()
    setup_project()
    fetch_data()
    model, tokenizer, device, val_loader = train_model()
    evaluate_model(model, tokenizer, device, val_loader)
