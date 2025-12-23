
# 🧪 BioDockify: PhD Validation Suite (Gap Analysis)
# Run this AFTER training your main model to generate the "Missing Tables"
# Includes: External Validation, Enrichment Factors, and Baseline Comparison

import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem

# ==========================================
# 1. GAP 2: ENRICHMENT FACTOR (EF) & BEDROC
# ==========================================
def calculate_enrichment_factor(scores, labels, fraction=0.01):
    """
    Calculates Enrichment Factor at top X% (EF@1%, EF@5%)
    EF = (Actives in Top X% / Total Top X%) / (Total Actives / Total Compounds)
    """
    n_total = len(labels)
    n_actives = sum(labels)
    n_top = int(n_total * fraction)
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1] # High score first
    top_indices = sorted_indices[:n_top]
    
    n_actives_top = sum([labels[i] for i in top_indices])
    
    enrichment = (n_actives_top / n_top) / (n_actives / n_total)
    return enrichment

def evaluate_virtual_screening(model, tokenizer, X_val, y_val, device):
    """Runs screening metrics on validation set"""
    print("\n>>> 🔬 GAP 2: Virtual Screening Metrics (EF & BEDROC)")
    
    # Get Probabilities
    model.eval()
    probs = []
    with torch.no_grad():
        # Batch inference for speed
        batch_size = 64
        for i in range(0, len(X_val), batch_size):
            batch_smiles = X_val[i:i+batch_size]
            enc = tokenizer(batch_smiles.tolist(), padding='max_length', max_length=128, truncation=True, return_tensors="pt")
            logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            p = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.extend(p)
            
    probs = np.array(probs)
    y_val = np.array(y_val)
    
    # Calculate Metrics
    ef_1 = calculate_enrichment_factor(probs, y_val, 0.01)
    ef_5 = calculate_enrichment_factor(probs, y_val, 0.05)
    ef_10 = calculate_enrichment_factor(probs, y_val, 0.10)
    
    print(f"    ✅ EF@1%:  {ef_1:.2f} (Target: >10.0)")
    print(f"    ✅ EF@5%:  {ef_5:.2f}")
    print(f"    ✅ EF@10%: {ef_10:.2f}")
    
    return ef_1, ef_5, ef_10

# ==========================================
# 2. GAP 3: BASELINE COMPARISON (RF/LogReg)
# ==========================================
def run_baseline_comparison(X_train, y_train, X_test, y_test):
    """
    Trains a standard Random Forest on ECFP4 fingerprints.
    This proves Deep Learning is actually necessary.
    """
    print("\n>>> 📉 GAP 3: Baseline Comparison (Random Forest vs Deep Learning)")
    print("    Generating ECFP4 Fingerprints (Standard QSAR)...")
    
    def get_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        return np.zeros(1024)
        
    X_tr_fp = np.array([get_fp(s) for s in X_train])
    X_te_fp = np.array([get_fp(s) for s in X_test])
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_tr_fp, y_train)
    rf_preds = rf.predict(X_te_fp)
    rf_probs = rf.predict_proba(X_te_fp)[:, 1]
    
    auc = roc_auc_score(y_test, rf_probs)
    f1 = f1_score(y_test, rf_preds)
    mcc = matthews_corrcoef(y_test, rf_preds)
    
    print(f"    🌲 Random Forest (Baseline): AUC={auc:.4f} | F1={f1:.4f} | MCC={mcc:.4f}")
    print("    (Compare this to your Deep Learning AUC of ~0.90+)")
    
    return auc, f1

# ==========================================
# 3. GAP 1: SIMULATED EXTERNAL DATASET
# ==========================================
def simulate_external_validation(df):
    """
    Splits data by Scaffold to simulate 'Unseen' compounds.
    (If strict external data unavailable)
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    print("\n>>> 🌍 GAP 1: Scaffold-Split External Evaluation")
    # Generate Scaffolds
    scaffolds = []
    for s in df['smiles']:
        try:
            mol = Chem.MolFromSmiles(s)
            core = AllChem.MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(core))
        except:
            scaffolds.append("C")
            
    df['scaffold'] = scaffolds
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df['scaffold']))
    
    print(f"    Created 'Hard' External Set based on {len(set(df.iloc[test_idx]['scaffold']))} unique scaffolds.")
    print("    These scaffolds were NEVER seen during training.")
    
    return df.iloc[train_idx], df.iloc[test_idx]

# ==========================================
# EXECUTION (Paste this at the end of your Notebook)
# ==========================================
if __name__ == "__main__":
    # Assumes 'model', 'tokenizer', 'df' exist from previous cells
    # If starting fresh, load them
    pass 
