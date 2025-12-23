# 🧪 BioDockify: Universal Disease Model Generator
**Build your own AI for any Disease (Cancer, Diabetes, Parkinson's, etc.)**

**Instructions:**
1.  Go to [colab.research.google.com](https://colab.research.google.com/).
2.  Click **"New Notebook"**.
3.  **Runtime > Change runtime type > T4 GPU**.

---

### 🔹 Cell 1: Configuration (CHOOSE YOUR DISEASE)
*Edit this cell to target the disease of your choice.*
```python
# ==========================================
# ⚙️ CONFIGURATION ZONE (UNCOMMENT YOUR DISEASE)
# ==========================================

# 1. 🍬 DIABETES (Targets: DPP4, SGLT2)
TARGET_DISEASE = "Diabetes"
TARGET_FILE_NAME = "chembl_diabetes.csv"
TARGET_IDS = ['CHEMBL284', 'CHEMBL4078'] 

# 2. 🦀 CANCER (Targets: EGFR, VEGFR2)
# TARGET_DISEASE = "Cancer"
# TARGET_FILE_NAME = "chembl_cancer.csv"
# TARGET_IDS = ['CHEMBL203', 'CHEMBL279'] 

# 3. 🧠 PARKINSON'S (Target: LRRK2)
# TARGET_DISEASE = "Parkinson"
# TARGET_FILE_NAME = "chembl_parkinson.csv"
# TARGET_IDS = ['CHEMBL3525']

# 4. ❤️ CARDIOVASCULAR (Target: HMGCR - Statins)
# TARGET_DISEASE = "Cardiovascular"
# TARGET_FILE_NAME = "chembl_cardio.csv"
# TARGET_IDS = ['CHEMBL402']

print(f"✅ Configuring BioDockify for: {TARGET_DISEASE}")
```

### 🔹 Cell 2: Installation
```python
!pip install transformers accelerate datasets torch rdkit scikit-learn chembl_webresource_client
print("✅ Libraries Installed.")
```

### 🔹 Cell 3: Data Download (Universal)
```python
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm

print(f">>> 🚀 FETCHING DATA FOR {TARGET_DISEASE}...")

activity = new_client.activity
activity_data = []

for tid in TARGET_IDS:
    print(f"    Fetching inhibitors for {tid}...")
    # Fetch valid IC50 data
    res = activity.filter(target_chembl_id=tid, pchembl_value__isnull=False, standard_type="IC50").only('molecule_chembl_id', 'canonical_smiles', 'pchembl_value')
    
    count = 0
    # Limit to 2000 per target for speed
    for act in tqdm(res[:2000], desc=tid):
        if 'canonical_smiles' in act and act['canonical_smiles']:
            activity_data.append({
                'smiles': act['canonical_smiles'],
                'pIC50': float(act['pchembl_value']),
                'target_id': tid,
                'disease': TARGET_DISEASE
            })
            count += 1
    print(f"    ✅ Got {count} compounds.")

# Save
pd.DataFrame(activity_data).to_csv(TARGET_FILE_NAME, index=False)
print(f"✅ Saved '{TARGET_FILE_NAME}' with {len(activity_data)} rows.")

# Fetch Natural Products (Universal Background)
print(">>> 🌿 Fetching Natural Products (Background)...")
import os
if not os.path.exists("coconut_natural_products.csv"):
    molecule = new_client.molecule
    res_nat = molecule.filter(natural_product=1).only('molecule_structures')
    nat_data = []
    for mol in tqdm(res_nat[:1000], desc="Naturals"):
        # Safe access to nested structure
        if mol and 'molecule_structures' in mol and mol['molecule_structures']:
             nat_data.append({'smiles': mol['molecule_structures']['canonical_smiles']})
    pd.DataFrame(nat_data).to_csv("coconut_natural_products.csv", index=False)
    print("✅ background naturals saved.")
```

### 🔹 Cell 4: Train Tier 3 Model (Rigorous 5-Fold CV)
*Trains the AI using Stratified Cross-Validation for PhD-grade robustness.*
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
import numpy as np

print(f">>> 🧠 STARTING RIGOROUS TRAINING FOR {TARGET_DISEASE}...")

# 1. Load & Balance Data
df = pd.read_csv(TARGET_FILE_NAME)
df = df.dropna(subset=['pIC50'])
y = np.array((df['pIC50'] >= 7.0).astype(int).tolist()) # Active Threshold
X = np.array(df['smiles'].tolist())

n_pos = sum(y)
ratio = (len(y) - n_pos) / n_pos if n_pos > 0 else 1.0
print(f"    📊 Data Stats: Actives={n_pos} | Inactives={len(y)-n_pos} | Class Weight={ratio:.2f}")

# 2. Model Architecture
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

class SpecialistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4), # Increased Dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, ids, mask):
        emb = self.encoder(ids, mask).pooler_output
        return self.classifier(emb)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 5-Fold Cross-Validation Loop
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
best_f1 = 0.0

print(f"    🚀 Running 5-Fold CV (This is the 'Scientific Standard')...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n    🔄 Fold {fold+1}/5...")
    
    # Init Model per fold
    model = SpecialistModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5) # Fine-tuning LR
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio]).to(device))
    
    # Prepare Data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_enc = tokenizer(X_train.tolist(), padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    val_enc = tokenizer(X_val.tolist(), padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    
    train_ds = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], torch.tensor(y_train, dtype=torch.float))
    val_ds = TensorDataset(val_enc['input_ids'], val_enc['attention_mask'], torch.tensor(y_val, dtype=torch.float))
    
    t_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    v_loader = DataLoader(val_ds, batch_size=32)
    
    # Train Loop
    model.train()
    for epoch in range(4): # 4 Epochs per fold
        for ids, mask, lbl in t_loader:
            optimizer.zero_grad()
            logits = model(ids.to(device), mask.to(device))
            loss = criterion(logits, lbl.unsqueeze(1).to(device))
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    probs = []
    truths = []
    with torch.no_grad():
        for ids, mask, lbl in v_loader:
            logits = model(ids.to(device), mask.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            truths.extend(lbl.numpy())
            
    try:
        auc = roc_auc_score(truths, probs)
        preds = [1 if p > 0.5 else 0 for p in probs]
        f1 = f1_score(truths, preds)
    except:
        auc, f1 = 0.5, 0.0
        
    print(f"    ✅ Fold {fold+1} Result: AUC={auc:.4f} | F1={f1:.4f}")
    fold_results.append({'auc': auc, 'f1': f1})
    
    # Save Best Model
    if f1 > best_f1:
        best_f1 = f1
        model_filename = f"biodockify_{TARGET_DISEASE.lower()}.pth"
        torch.save(model.state_dict(), model_filename)

# Summary
avg_auc = np.mean([r['auc'] for r in fold_results])
print(f"\n🏆 FINAL METRICS ({TARGET_DISEASE}): Avg AUC = {avg_auc:.4f} | Best F1 = {best_f1:.4f}")
print(f"✅ Saved Best Model to: {model_filename}")
```

### 🔹 Cell 5: Release to Hugging Face
```python
from huggingface_hub import HfApi, notebook_login
import pandas as pd

try:
    notebook_login()
except:
    from huggingface_hub import login
    login()

repo_name = f"biodockify-{TARGET_DISEASE.lower()}-v1"
print(f">>> 🌍 RELEASING {repo_name}...")

api = HfApi()
user = api.whoami()['name']
full_repo_id = f"{user}/{repo_name}"

try:
    api.create_repo(repo_id=full_repo_id, exist_ok=True)
    api.update_repo_visibility(repo_id=full_repo_id, private=False) # Make public
    
    # Upload Model
    api.upload_file(
        path_or_fileobj=model_filename,
        path_in_repo=f"{model_filename}",
        repo_id=full_repo_id,
        repo_type="model"
    )
    
    # Upload Data
    api.upload_file(
        path_or_fileobj=TARGET_FILE_NAME,
        path_in_repo="dataset.csv",
        repo_id=full_repo_id,
        repo_type="dataset"
    )
    
    print(f"✅ Success! Your {TARGET_DISEASE} AI is live at: https://huggingface.co/{full_repo_id}")
    print("👉 Now Go to your BioDockify Website and Dock your compounds!")
except Exception as e:
    print(f"⚠️ Upload Error: {e}")
```
