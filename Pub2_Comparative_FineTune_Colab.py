# =============================================================================
# TIER 2: Comparative Phytochemical-Pharmacological Profiling (Fine-Tuning)
# Title: Multi-Target Alzheimer's Lead Discovery of various medicinal plants extracts
# Target Journal: Journal of Cheminformatics / Phytomedicine
# Architecture: Domain-Adapted ChemBERTa + MolFormer Ensemble
# =============================================================================

# 1. SETUP & INSTALLATION
# !pip install -q torch==2.1.0 transformers==4.36.0 datasets==2.14.6 \
#     deepchem==2.7.1 rdkit-pypi==2023.9.6 scikit-learn==1.3.2 \
#     shap==0.45.0 accelerate==0.25.0 seaborn matplotlib

import torch
import numpy as np
import pandas as pd
import deepchem as dc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, r2_score
from datasets import Dataset
import shap
import matplotlib.pyplot as plt
import seaborn as sns

print(f"GPU Available: {torch.cuda.is_available()}")

# =============================================================================
# 2. DATA CURATION: The "Comparative" Datasets
# =============================================================================

# A. Domain Adaptation Dataset (Natural Products)
# In a real run, load COCONUT or a subset of flavonoids/alkaloids
# Here we simulate with a list of common natural scaffolds
natural_scaffolds = [
    "c1cc(O)cc(O)c1", # Resorcinol substructure
    "O=C1C=C(O)CC(=O)C1",
    "C1=CC=C(C=C1)C=CC(=O)O", # Cinnamic acid derivative
    "OC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(O)C=C3O" # Apigenin backbone
] * 500 # Replicate to simulate volume

# B. Target Dataset (Alzheimer's Inhibitors from ChEMBL)
# Simulated ChEMBL data
chembl_smiles = [
    "CC1=CC=C(C=C1)C(=O)Nc2ccc(cc2)S(=O)(=O)N3CCN(CC3)C", # Donepezil-like
    "COc1ccc(cc1OC)N2CCN(CC2)C(=O)Cc3ccc(OCCN4CCOCC4)cc3" # Galantamine-like
] * 2000

# C. YOUR Comparative Extracts (Placeholders - Replace with LC-MS Lists)
plant_a_smiles = [ # Evolvulus alsinoides
    "CC1=CC(=O)C2=C(C1=O)C(=CC=C2O)O", # Quercetin
    "C1=CC(=C(C(=C1)O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O", # Rutin
    "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin (Control)
]

plant_b_smiles = [ # Cordia dichotoma
    "CC1=CC(=O)C2=CC(=C(C=C2C1=O)O)O", # Kaempferol
    "COC1=C(C=CC(=C1)O)C(CC(=O)C2=CC(=C(C=C2)O)O)O",
    "CCCCCCCC(=O)O"
]

print(f"Data Loaded: {len(natural_scaffolds)} NP-Scaffolds, {len(chembl_smiles)} ChEMBL Inhibitors")

# Combine for Training (Domain Adaptation + Target Tuning)
# Strategy: Train on everything, but prioritize NP-like molecules?
# Simpler Strategy: Single training set with all available valid data.
full_train_smiles = natural_scaffolds + chembl_smiles
# Synthetic labels (replace with real pIC50s)
labels = np.concatenate([
    np.random.normal(5.0, 1.0, len(natural_scaffolds)), # NPs might be inactive on avg
    np.random.normal(7.5, 1.2, len(chembl_smiles))      # Actives are higher
])

dataset = Dataset.from_dict({"smiles": full_train_smiles, "labels": labels})
dataset = dataset.train_test_split(test_size=0.1)

# =============================================================================
# 3. FINE-TUNING: ChemBERTa w/ Domain focus
# =============================================================================

model_name = "DeepChem/ChemBERTa-2-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

def tokenize_function(examples):
    return tokenizer(examples["smiles"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# =============================================================================
# 3. ADVANCED FINE-TUNING: Similarity-Weighted Loss (Plant Specificity)
# =============================================================================

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def compute_similarity_weights(train_smiles, target_plant_smiles):
    """
    Calculates weights for training samples based on their structural similarity
    to the target plant extracts. Higher similarity = Higher Weight.
    """
    print("Computing Tanimoto Similarity Weights...")
    # 1. Generate Fingerprints for Target Plant (Your Extracts)
    target_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2) 
                  for s in target_plant_smiles if Chem.MolFromSmiles(s)]
    
    weights = []
    for s in train_smiles:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            weights.append(1.0)
            continue
            
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        
        # Max similarity to ANY of your plant compounds
        sims = DataStructs.BulkTanimotoSimilarity(fp, target_fps)
        max_sim = max(sims) if sims else 0.0
        
        # Weight Formula: Base 1.0 + (Similarity * 10.0)
        # If identical (1.0), weight is 11.0. If distinct (0.0), weight is 1.0.
        weights.append(1.0 + (max_sim * 10.0))
        
    return torch.tensor(weights, dtype=torch.float32)

# Calculate Weights for the Training Set
# We focus finding matches to Evolvulus & Cordia
all_plant_targets = plant_a_smiles + plant_b_smiles
train_weights = compute_similarity_weights(dataset["train"]["smiles"], all_plant_targets)

# Custom Trainer to apply these weights
class PlantSpecificTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Standard MSE Loss
        loss_fct = torch.nn.MSELoss(reduction='none')
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        
        # Apply Instance Weights (Must align with current batch)
        # Note: In a real script, we'd pass weights via the dataset/collator.
        # For this simplified Colab version, we map weights via sample index or approximation.
        # Here we assume unweighted for 'simple' run, or we'd need a CustomDataCollator.
        # IMPLEMENTATION NOTE: To make this robust in Colab, we add weights to dataset
        
        return (loss.mean(), outputs) if return_outputs else loss.mean()

# For the Colab script, we will just use the standard Trainer but with the 
# "Plant-Like" data oversampled in the dataset input generation which is easier/safer.

# REVISED STRATEGY: Data Augmentation via Similarity
# Instead of complex custom loss, we CLONE high-similarity data 5x.
print("Applying Similarity Augmentation (Oversampling Plant-Like Compounds)...")
aug_smiles = []
aug_labels = []

train_smiles_list = dataset["train"]["smiles"]
train_labels_list = dataset["train"]["labels"]
w = train_weights.numpy()

for i, weight in enumerate(w):
    # If similarity score > 0.4 (approx weight > 5), duplicate it
    if weight > 5.0: 
        aug_smiles.extend([train_smiles_list[i]] * 5)
        aug_labels.extend([train_labels_list[i]] * 5)
    else:
        aug_smiles.append(train_smiles_list[i])
        aug_labels.append(train_labels_list[i])

train_dataset_augmented = Dataset.from_dict({"smiles": aug_smiles, "labels": aug_labels})
tokenized_train = train_dataset_augmented.map(tokenize_function, batched=True)

# Training Arguments
args = TrainingArguments(
    output_dir="./bio_dockify_pub2_model",
    num_train_epochs=5, # Increased epochs for specialization
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5, # Lower LR for fine-tuning
    weight_decay=0.01,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=lambda p: {"rmse": mean_squared_error(p.label_ids, p.predictions, squared=False)}
)

print("Starting Plant-Specific Fine-Tuning...")
trainer.train()
trainer.save_model("biodockify_pub2_model")

# =============================================================================
# 4. COMPARATIVE INFERENCE (The Result Generation)
# =============================================================================

def predict_pIC50(smiles_list):
    # Quick inference loop
    inputs = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy().flatten()
    return logits

print("\n--- Generating Comparative Profile ---")
preds_a = predict_pIC50(plant_a_smiles)
preds_b = predict_pIC50(plant_b_smiles)

df_results = pd.DataFrame({
    "Plant": ["Evolvulus"]*len(preds_a) + ["Cordia"]*len(preds_b),
    "SMILES": plant_a_smiles + plant_b_smiles,
    "Predicted_pIC50": np.concatenate([preds_a, preds_b])
})

# A. Statistical Comparison
print(df_results.groupby("Plant")["Predicted_pIC50"].describe())

# B. Visual Comparison (for Paper)
plt.figure(figsize=(8,6))
sns.violinplot(x="Plant", y="Predicted_pIC50", data=df_results, inner="stick")
plt.title("Comparative Pharmacological Profile (Predicted Affinity)")
plt.axhline(6.0, ls='--', color='red', label='Active Threshold')
plt.legend()
plt.savefig("Figure_Comparative_Profile.png")
print("Saved Figure_Comparative_Profile.png")

# C. Top Hits Export
top_hits = df_results[df_results["Predicted_pIC50"] > 6.0].sort_values("Predicted_pIC50", ascending=False)
top_hits.to_csv("Comparative_Top_Hits.csv", index=False)
print("Saved Comparative_Top_Hits.csv")
