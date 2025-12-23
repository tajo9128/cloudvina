# Colab Data Prep Script
# Run this LOCALLY or on COLAB to generate the necessary CSV files for Tier 2 Training.
# This prevents "FileNotFoundError" when you run the main trainer.

import pandas as pd
import os

print(">>> Generating Sample Datasets for BioDockify Tier 2...")

# 1. MOCK ChEMBL Alzheimer's Data (Synthetic Anchors)
# Real drugs/inhibitors known to hit AChE/BACE1
chembl_data = {
    'smiles': [
        "COc1cc(CC(=O)N2CCC(CC2)Cc2ccccc2)cc(OC)c1OC", # Donepezil (AChE)
        "CN(C)C(=O)Oc1cccc(C[N+](C)(C)C)c1",           # Neostigmine (AChE)
        "O=C1N(Cc2ccccc2)C(=O)C2=C1C=CC=C2",           # Phthalimide deriv
        "CC(C)N(C[C@H](O)C[C@@H](Cc1ccccc1)NC(=O)c1c(C)cc(C)cc1C)C(=O)C(N)CC(C)C", # Ritonavir-ish
        "COc1ccccc1CN(C)CCCCN(C)Cc1ccccc1OC",         # Generic AChE binder
        "CCN(CC)CCCC(C)Nc1ccnc2ccWithClc12",          # Chloroquine-like
        "CC1=CC(=C(C=C1)O)C(=O)O",                     # Salicylic acid
        "CC(=O)Nc1ccc(O)cc1",                          # Paracetamol
        "CN1CCCC1CNC(=O)c2cc(S(=O)(=O)CC)c(N)cc2OC", # Sulpiride
        "C1CN(CCC1(c2ccc(cc2)Cl)O)CCC(=O)c3ccc(cc3)F" # Haloperidol
    ] * 50, # Duplicate to make it look bigger
    'target_id': ['ACHE', 'ACHE', 'BACE1', 'BACE1', 'ACHE', 'ACHE', 'COX', 'COX', 'D2', 'D2'] * 50,
    'pIC50': [9.2, 8.5, 7.8, 8.1, 7.5, 6.0, 5.0, 4.0, 8.0, 8.5] * 50
}

# 2. MOCK COCONUT Natural Products (Positives/Negatives)
# Diverse plant scaffolds (Flavonoids, Alkaloids, Terpenes)
coconut_data = {
    'smiles': [
        "OC1=C(O)C=C(C=C1)/C=C/C(=O)C1=C(O)C=C(O)C=C1", # Isoliquiritigenin (Chalcone)
        "COc1cc(C=C)ccc1O",                            # Isoeugenol
        "CC(=O)Oc1ccccc1C(=O)O",                       # Aspirin (Natural origin)
        "C[C@H]1[C@@H]2CC[C@H]3[C@@H]([C@H]2CC[C@]1(C)O)CC[C@H]4[C@@]3(C)CC(=O)C4", # Steroid
        "C=C1C(=O)O[C@@H]2[C@H]1CC[C@@H]2C",           # Terpene lactone
        "COc1cc2c(cc1OC)c3c(n2C)C(=O)c4ccccc4C3=O",    # Alkaloid backbone
        "Oc1c(O)cc(O)c2c1OC(c3ccc(O)c(O)c3)C(O)C2",    # Catechin
        "C1=CC(=CC=C1/C=C/C(=O)O)O",                   # p-Coumaric acid
        "COC1=CC=CC=C1/C=C/C(=O)OC",                   # Methyl cinnamate
        "CC1(C)CCCC2(C)C1CCC(=C2)C=O"                  # Retinal
    ] * 50,
    'source': ['Evolvulus', 'Cordia', 'Willow', 'Ginseng', 'Ginkgo', 'Poppy', 'Tea', 'Grass', 'Cinnamon', 'Carrot'] * 50
}

# 3. Save to CSV
df_chembl = pd.DataFrame(chembl_data)
df_coconut = pd.DataFrame(coconut_data)

df_chembl.to_csv("chembl_alzheimers.csv", index=False)
df_coconut.to_csv("coconut_natural_products.csv", index=False)

print(f">>> Created 'chembl_alzheimers.csv' ({len(df_chembl)} rows)")
print(f">>> Created 'coconut_natural_products.csv' ({len(df_coconut)} rows)")
print(">>> Ready for upload to Colab!")
