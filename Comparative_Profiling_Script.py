# =============================================================================
# TIER 2: Comparative Phytochemical Profiling (Colab Pro Script)
# Title: Comparative Phytochemical-Pharmacological Profiling Through AI-Integrated CADD
# Plants: Evolvulus alsinoides vs Cordia dichotoma
# Dec 14-20, 2025 | Publication: JCIM/Phytomedicine
# =============================================================================

# 0. SETUP
import torch, gc; torch.cuda.empty_cache()
try:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
except:
    print("GPU not detected. Use Colab GPU Runtime.")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import deepchem as dc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. DEFINE PLANT DATASETS (The "Comparative" Part)
# Ideally, replace these list placeholders with your actual LC-MS SMILES lists
evolvulus_smiles = [
    "CC1=CC(=O)C2=C(C1=O)C(=CC=C2O)O", # Quercetin
    "C1=CC(=C(C(=C1)O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O", # Rutin
    "CN1CC2=CC=CC=C2C1C3=CC=CC=C3", # Example Alkaloid
    # ... Paste full list here
]

cordia_smiles = [
    "CC1=CC(=O)C2=CC(=C(C=C2C1=O)O)O", # Kaempferol
    "COC1=C(C=CC(=C1)O)C(CC(=O)C2=CC(=C(C=C2)O)O)O", # Example Flavonoid
    "CCCCCCCC(=O)O", # Fatty Acid
    # ... Paste full list here
]

# Standard Drugs (Reference)
reference_smiles = [
    "CC1=CC=C(C=C1)C(=O)Nc2ccc(cc2)S(=O)(=O)N3CCN(CC3)C", # Donepezil
    "COc1ccc(cc1OC)N2CCN(CC2)C(=O)Cc3ccc(OCCN4CCOCC4)cc3", # Galantamine
    "CN(C)C(=O)Oc1cccc(c1)[C@@H](C)N(C)C" # Rivastigmine
]

# 2. LOAD & FINE-TUNE MODEL (Condensed for brevity - use full script for actual training)
print("Loading Model (Assume Fine-Tuned Checkpoint Loaded)...")
model_path = "biodockify/chemberta-ad-v2" # Or local path
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-2-77M-MTR")
# In real run: Load actual fine-tuned model
# model = AutoModelForSequenceClassification.from_pretrained(model_path) 

# Mock Predict Function (Replace with actual Trainer.predict)
def predict_activity(smiles_list):
    # Mocking predictions for demonstration: Gaussian dist around typical means
    # Real code would tokenize and pass through model
    return np.random.normal(7.0, 1.5, len(smiles_list))

# 3. RUN COMPARATIVE SCREEN
print("Running Comparative AI Screen...")
results = []

# Evolvulus
scores_e = predict_activity(evolvulus_smiles)
for s, score in zip(evolvulus_smiles, scores_e):
    results.append({'Plant': 'Evolvulus alsinoides', 'SMILES': s, 'pIC50': score, 'Type': 'Extract'})

# Cordia
scores_c = predict_activity(cordia_smiles)
for s, score in zip(cordia_smiles, scores_c):
    results.append({'Plant': 'Cordia dichotoma', 'SMILES': s, 'pIC50': score, 'Type': 'Extract'})

# Reference
scores_r = predict_activity(reference_smiles)
for s, score in zip(reference_smiles, scores_r):
    results.append({'Plant': 'Reference (Drugs)', 'SMILES': s, 'pIC50': score, 'Type': 'Standard'})

df = pd.DataFrame(results)

# 4. GENERATE COMPARATIVE PLOTS (The "Output" for Paper)

# A. Comparative Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Plant', y='pIC50', data=df, palette="Set3")
plt.axhline(y=6.0, color='r', linestyle='--', label='Active Threshold')
plt.title("Comparative Predicted Anti-Alzheimer's Potential")
plt.ylabel("Predicted pIC50 (Affinity)")
plt.savefig("Comparative_Profile_Boxplot.png")
print("Saved Comparative_Profile_Boxplot.png")

# B. Heatmap of Top Isolates
top_e = df[df['Plant']=='Evolvulus alsinoides'].nlargest(5, 'pIC50')
top_c = df[df['Plant']=='Cordia dichotoma'].nlargest(5, 'pIC50')
top_overall = pd.concat([top_e, top_c])

plt.figure(figsize=(12, 6))
pivot = top_overall.pivot_table(index='SMILES', columns='Plant', values='pIC50')
sns.heatmap(pivot, annot=True, cmap='viridis', cbar_kws={'label': 'pIC50'})
plt.title("Heatmap of Top Isolated Compounds")
plt.savefig("Top_Isolates_Heatmap.png")
print("Saved Top_Isolates_Heatmap.png")

# 5. EXPORT RESULTS
df.to_csv("Comparative_Phytochemical_Profile.csv", index=False)
print("Saved Comparative_Phytochemical_Profile.csv")
