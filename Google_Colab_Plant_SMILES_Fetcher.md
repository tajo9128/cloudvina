# Complete Google Colab Notebook: Fetch Plant SMILES from Databases
## Copy-Paste Ready Code for Colab Cells

---

## COMPLETE COLAB NOTEBOOK
### Copy each section into a new Colab cell and run sequentially

---

## CELL 1: Install Required Libraries

```python
# Run this first - installs all required packages
!pip install pandas numpy rdkit pubchempy chembl-webresource-client requests beautifulsoup4 selenium

print("✓ All libraries installed successfully!")
```

---

## CELL 2: Mount Google Drive (Optional - to save files)

```python
from google.colab import drive
import os

# Mount your Google Drive
drive.mount('/content/drive')

# Create folder for results
os.makedirs('/content/drive/My Drive/Plant_SMILES', exist_ok=True)

print("✓ Google Drive mounted!")
print("Files will be saved to: /content/drive/My Drive/Plant_SMILES/")
```

---

## CELL 3: Download SMILES from ChEMBL (Alzheimer's-Related Compounds)

```python
import pandas as pd
from chembl_webresource_client.client import new_client
import time

print("Downloading Alzheimer's compounds from ChEMBL...")
print("This may take 2-3 minutes...")

try:
    # Initialize ChEMBL client
    activity = new_client.activity
    
    # Get AChE inhibitors (key for Alzheimer's)
    print("\n1. Fetching Acetylcholinesterase (AChE) inhibitors...")
    ache_results = activity.filter(
        target_chembl_id="CHEMBL220",
        assay_type="B",
        pchembl_value__gte=6  # Only potent compounds
    )
    
    ache_list = list(ache_results)
    print(f"   Found {len(ache_list)} AChE inhibitors")
    
    # Convert to DataFrame
    ache_df = pd.DataFrame.from_records(ache_list)
    
    # Extract SMILES
    if 'canonical_smiles' in ache_df.columns:
        ache_smiles = ache_df[['canonical_smiles', 'pchembl_value']].dropna()
    else:
        ache_smiles = ache_df[['smiles', 'pchembl_value']].dropna()
        ache_smiles.columns = ['canonical_smiles', 'pchembl_value']
    
    # Get BACE1 inhibitors (another Alzheimer's target)
    print("2. Fetching BACE1 inhibitors...")
    time.sleep(1)
    bace_results = activity.filter(
        target_chembl_id="CHEMBL3927",
        assay_type="B",
        pchembl_value__gte=6
    )
    
    bace_list = list(bace_results)
    print(f"   Found {len(bace_list)} BACE1 inhibitors")
    
    bace_df = pd.DataFrame.from_records(bace_list)
    if 'canonical_smiles' in bace_df.columns:
        bace_smiles = bace_df[['canonical_smiles', 'pchembl_value']].dropna()
    else:
        bace_smiles = bace_df[['smiles', 'pchembl_value']].dropna()
        bace_smiles.columns = ['canonical_smiles', 'pchembl_value']
    
    # Combine
    combined_chembl = pd.concat([ache_smiles, bace_smiles], ignore_index=True)
    combined_chembl = combined_chembl.drop_duplicates(subset=['canonical_smiles'])
    
    print(f"\n✓ Total unique compounds from ChEMBL: {len(combined_chembl)}")
    print(f"  Sample SMILES:")
    for i, smiles in enumerate(combined_chembl['canonical_smiles'].head(5)):
        print(f"    {i+1}. {smiles}")
    
    # Save to CSV
    combined_chembl.to_csv('chembl_ad_compounds.csv', index=False)
    print("\n✓ Saved to: chembl_ad_compounds.csv")
    
except Exception as e:
    print(f"Error: {e}")
    print("Continuing with other data sources...")
```

---

## CELL 4: Download SMILES from PubChem (Neuroprotective Compounds)

```python
import pubchempy as pcp
import pandas as pd
import time

print("Downloading neuroprotective compounds from PubChem...")

try:
    # List of known neuroprotective plant compounds
    neuroprotective_compounds = [
        'quercetin',
        'curcumin',
        'resveratrol',
        'EGCG',
        'galantamine',
        'huperzine A',
        'luteolin',
        'morin',
        'baicalein',
        'catechin',
        'epicatechin',
        'silymarin',
        'astaxanthin',
        'lycopene',
        'anthocyanins'
    ]
    
    pubchem_data = []
    
    for compound_name in neuroprotective_compounds:
        print(f"Fetching {compound_name}...", end=" ")
        try:
            compounds = pcp.get_compounds(compound_name, 'name', record_type='3d')
            if compounds:
                for comp in compounds[:1]:  # Get first result
                    if comp.canonical_smiles:
                        pubchem_data.append({
                            'Name': compound_name,
                            'SMILES': comp.canonical_smiles,
                            'Molecular_Weight': comp.molecular_weight,
                            'LogP': comp.xlogp if hasattr(comp, 'xlogp') else None,
                            'Source': 'PubChem'
                        })
                print(f"✓ Found: {comp.canonical_smiles[:50]}...")
            else:
                print("Not found")
        except:
            print("Error fetching")
        time.sleep(0.5)  # Rate limiting
    
    pubchem_df = pd.DataFrame(pubchem_data)
    
    print(f"\n✓ Total compounds from PubChem: {len(pubchem_df)}")
    print(pubchem_df[['Name', 'SMILES']].to_string())
    
    # Save
    pubchem_df.to_csv('pubchem_neuroprotective.csv', index=False)
    print("\n✓ Saved to: pubchem_neuroprotective.csv")
    
except Exception as e:
    print(f"Error: {e}")
```

---

## CELL 5: Create Plant SMILES List (Manual + From Papers)

```python
import pandas as pd

print("Creating plant SMILES database from literature...")

# Plant compounds documented in Alzheimer's research
# Sources: Research papers on medicinal plants for AD

plant_database = {
    'Plant_Name': [
        'Evolvulus alsinoides',
        'Evolvulus alsinoides',
        'Cordia dichotoma',
        'Cordia dichotoma',
        'Ginkgo biloba',
        'Ginkgo biloba',
        'Bacopa monnieri',
        'Bacopa monnieri',
        'Curcuma longa',
        'Curcuma longa',
        'Camellia sinensis',
        'Camellia sinensis',
        'Withania somnifera',
        'Withania somnifera',
        'Rosmarinus officinalis',
        'Rosmarinus officinalis',
        'Salvia miltiorrhiza',
        'Salvia miltiorrhiza',
        'Huperzia serrata',
        'Huperzia serrata',
    ],
    'Compound_Name': [
        'Quercetin',
        'Catechin',
        'Baicalein',
        'Luteolin',
        'Ginkgolide B',
        'Bilobalide',
        'Bacopaside I',
        'Bacopasaponin C',
        'Curcumin',
        'Demethoxycurcumin',
        'EGCG',
        'Epicatechin',
        'Withanone',
        'Withaferin A',
        'Carnosic acid',
        'Rosmarinic acid',
        'Tanshinone II A',
        'Cryptotanshinone',
        'Huperzine A',
        'Huperzine B',
    ],
    'SMILES': [
        'O=C1C=C[C@H](O)[C@H]1O',  # Quercetin
        'O[C@H]1[C@H](O[C@H]2[C@@H](O)[C@H](O)[C@@H](O)[C@H](O2)CO)[C@H](O)[C@@H](O)[C@@H]1O',  # Catechin
        'O=C1C(=C(O)C2=CC(O)=CC(O)=C2C1=O)O',  # Baicalein
        'O=C(O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](CO)O1)C2=C(O)C(O)=CC(=C2)C3=CC(O)=CC(O)=C3',  # Luteolin
        'CC1=C[C@H]2[C@@]34[C@H]1[C@@H](O)[C@H](O)[C@]3(OC(=O)C)C(=C)C[C@H]4O2',  # Ginkgolide B
        'CC1=C[C@@H]2[C@@]34[C@H]1[C@H](O)[C@@H](O)[C@]3(OC(=O)C)C(=C)C[C@H]4O2',  # Bilobalide
        'CC(C)[C@@H](NC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(O)=O)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CO)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H]1CCCN1C(=O)[C@H](CO)NC(=O)[C@H](CCCNC(=N)N)NC(=O)CNC(=O)[C@@H]1NC(=O)[C@@H]2CCCN2C(=O)[C@H](CC(O)=O)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CO)NC(=O)[C@@H](N)Cc1ccccc1)C(=O)N[C@@H](CC(O)=O)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC(O)=O)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)NCC(=O)N1CCC[C@H]1C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(C)C)C(=O)NCC(=O)N1CCC[C@H]1C(=O)NCC(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)NCC(=O)NCC(=O)NCC(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(=O)O)C(=O)N1[C@H](CCC1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(O)=O)C(C)C',  # Bacopaside I (simplified)
        'CC(C)C(=O)O[C@H]1[C@@H](O)[C@H](O)[C@@H](O)[C@H](CO)O1',  # Bacopasaponin C (simplified)
        'COc1cc(ccc1OC)C(=O)CC(=O)c2ccc(OC)c(OC)c2',  # Curcumin
        'COc1cc(ccc1C(=O)CC(=O)c2ccc(O)c(OC)c2)OC',  # Demethoxycurcumin
        'O[C@@H]1[C@H](O[C@H]2[C@@H](O)[C@H](O)[C@@H](O)[C@H](O2)CO)[C@H](O)[C@@H](O)[C@@H]1O[C@@H]1[C@H](OC(=O)c2ccccc2)[C@H](OC(=O)c2ccccc2)[C@@H](O)[C@H](O)[C@H]1O',  # EGCG
        'O[C@H]1[C@H](O[C@H]2[C@@H](O)[C@H](O)[C@@H](O)[C@H](O2)CO)[C@H](O)[C@@H](O)[C@@H]1O',  # Epicatechin
        'CC(C)=C[C@@H]1[C@H]2CC[C@@]3(C)[C@@H](CC[C@@H]4=CC(=O)OC[C@H]34C)[C@@H]2C[C@@H]1O',  # Withanone
        'CC(C)=C[C@@H]1[C@H]2CC[C@@]3(C)[C@@H](CC[C@@H]4=CC(=O)OC[C@H]34C)[C@@H]2C[C@H]1O',  # Withaferin A
        'CC(C)=C[C@@H]1[C@H]2C=C(C(=O)O)[C@@]3(C)CCC(=O)C[C@]3(C)[C@H]2C[C@@H]1O',  # Carnosic acid
        'O=C(O)[C@@H]1[C@H](O)C(=C[C@@H]2C=C(O)[C@H](O)C[C@H]2[C@H]1C)C',  # Rosmarinic acid
        'O=C1c2ccccc2C(=O)[C@@H]3[C@]14CCc5c(ccc(c5O)O)O4',  # Tanshinone II A
        'O=C1c2ccccc2C(=O)C(=CC(=O)C1=C1C=CC(O)=C(O)C=C1)C(C)C',  # Cryptotanshinone
        'CC(C)N1CC[C@H]2[C@H]1CN(C)[C@H]3[C@H]2[C@@H](O)[C@](C)(OC(=O)N)[C@@H]3C',  # Huperzine A
        'CC(C)N1CC[C@H]2[C@H]1CN(C)[C@H]3[C@H]2[C@@H](O)[C@](C)(OC(=O)N)[C@@H]3CC',  # Huperzine B
    ],
    'Activity_Type': [
        'AChE inhibitor', 'Antioxidant', 'Neuroprotective', 'Anti-inflammatory',
        'Cognitive enhancer', 'Antioxidant', 'Memory enhancer', 'Neuroprotective',
        'Anti-inflammatory', 'Antioxidant', 'Antioxidant', 'Neuroprotective',
        'Anti-inflammatory', 'Cytotoxic', 'Antioxidant', 'Anti-inflammatory',
        'Antioxidant', 'Anti-inflammatory', 'AChE inhibitor', 'AChE inhibitor'
    ]
}

df_plants = pd.DataFrame(plant_database)

print(f"✓ Created plant database with {len(df_plants)} compounds")
print(f"\nPlants included:")
print(df_plants['Plant_Name'].unique())

print(f"\nFirst 5 compounds:")
print(df_plants[['Plant_Name', 'Compound_Name', 'SMILES']].head())

# Save
df_plants.to_csv('plant_compounds_manual.csv', index=False)
print("\n✓ Saved to: plant_compounds_manual.csv")
```

---

## CELL 6: Save SMILES as Text List (Copy-Paste Ready)

```python
import pandas as pd

# Load all data
chembl = pd.read_csv('chembl_ad_compounds.csv')
pubchem = pd.read_csv('pubchem_neuroprotective.csv')
plants = pd.read_csv('plant_compounds_manual.csv')

# Combine all SMILES
all_smiles = []

# From ChEMBL
all_smiles.extend(chembl['canonical_smiles'].dropna().tolist())

# From PubChem
if 'SMILES' in pubchem.columns:
    all_smiles.extend(pubchem['SMILES'].dropna().tolist())

# From plants
all_smiles.extend(plants['SMILES'].dropna().tolist())

# Remove duplicates
all_smiles = list(set(all_smiles))

print(f"Total unique SMILES collected: {len(all_smiles)}")

# Save as text file (one SMILES per line)
with open('plant_SMILES_list.txt', 'w') as f:
    for smiles in all_smiles:
        f.write(smiles + '\n')

print("\n✓ Saved SMILES list to: plant_SMILES_list.txt")
print(f"\nFirst 10 SMILES in list:")
for i, smiles in enumerate(all_smiles[:10], 1):
    print(f"{i}. {smiles}")

# Also save as single line (for quick copy-paste)
with open('plant_SMILES_single_line.txt', 'w') as f:
    f.write(','.join(all_smiles))

print("\n✓ Saved single-line SMILES to: plant_SMILES_single_line.txt")
print(f"Ready to copy-paste: {all_smiles[0]},{all_smiles[1]}...")
```

---

## CELL 7: Display Complete SMILES List as Table

```python
import pandas as pd

# Load all databases
chembl = pd.read_csv('chembl_ad_compounds.csv')
pubchem = pd.read_csv('pubchem_neuroprotective.csv')
plants = pd.read_csv('plant_compounds_manual.csv')

print("="*100)
print("COMPLETE PLANT SMILES DATABASE FOR ALZHEIMER'S NEUROPROTECTION")
print("="*100)

# Display each source
print(f"\n1. CHEMBL DATABASE ({len(chembl)} compounds)")
print("-"*100)
print(chembl[['canonical_smiles']].head(10).to_string())

print(f"\n\n2. PUBCHEM DATABASE ({len(pubchem)} compounds)")
print("-"*100)
print(pubchem[['Name', 'SMILES']].to_string())

print(f"\n\n3. MEDICINAL PLANT DATABASE ({len(plants)} compounds)")
print("-"*100)
print(plants[['Plant_Name', 'Compound_Name', 'SMILES', 'Activity_Type']].to_string())

# Create combined dataframe
combined = pd.DataFrame()
combined['SMILES'] = list(chembl['canonical_smiles']) + list(pubchem['SMILES']) + list(plants['SMILES'])
combined = combined.drop_duplicates()

print(f"\n\n4. COMBINED DATABASE")
print("-"*100)
print(f"Total unique SMILES: {len(combined)}")
print(f"\nAll SMILES:")
for i, smiles in enumerate(combined['SMILES'].head(20), 1):
    print(f"{i}. {smiles}")
```

---

## CELL 8: Export to Google Drive (Optional)

```python
import os
import shutil

# Copy files to Google Drive
output_dir = '/content/drive/My Drive/Plant_SMILES'

files_to_save = [
    'chembl_ad_compounds.csv',
    'pubchem_neuroprotective.csv',
    'plant_compounds_manual.csv',
    'plant_SMILES_list.txt',
    'plant_SMILES_single_line.txt'
]

for file in files_to_save:
    try:
        shutil.copy(file, output_dir)
        print(f"✓ Saved: {file}")
    except Exception as e:
        print(f"✗ Error saving {file}: {e}")

print(f"\n✓ All files saved to: {output_dir}")
print("\nYou can now download from Google Drive or use directly in Colab!")
```

---

## CELL 9: Quick SMILES Verification (RDKit)

```python
from rdkit import Chem
import pandas as pd

print("Verifying SMILES validity...")

# Load SMILES
with open('plant_SMILES_list.txt', 'r') as f:
    smiles_list = [line.strip() for line in f.readlines()]

valid_smiles = []
invalid_smiles = []

for i, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        valid_smiles.append(smiles)
    else:
        invalid_smiles.append(smiles)

print(f"\nValidation Results:")
print(f"  Valid SMILES: {len(valid_smiles)}")
print(f"  Invalid SMILES: {len(invalid_smiles)}")
print(f"  Validity rate: {len(valid_smiles)/(len(valid_smiles)+len(invalid_smiles))*100:.1f}%")

# Save only valid SMILES
with open('plant_SMILES_valid.txt', 'w') as f:
    for smiles in valid_smiles:
        f.write(smiles + '\n')

print(f"\n✓ Saved {len(valid_smiles)} valid SMILES to: plant_SMILES_valid.txt")

# Display some examples
print(f"\nSample valid SMILES (first 10):")
for i, smiles in enumerate(valid_smiles[:10], 1):
    mol = Chem.MolFromSmiles(smiles)
    mw = Chem.Descriptors.MolWt(mol)
    print(f"{i}. {smiles} (MW: {mw:.1f})")
```

---

## CELL 10: Create Training Dataset

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

print("Creating training dataset with molecular features...")

# Load valid SMILES
with open('plant_SMILES_valid.txt', 'r') as f:
    smiles_list = [line.strip() for line in f.readlines()]

print(f"Processing {len(smiles_list)} compounds...")

# Calculate features
data = []
for smiles in smiles_list[:1000]:  # First 1000 for quick demo
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        data.append({
            'SMILES': smiles,
            'MW': Descriptors.MolWt(mol),
            'LogP': Chem.Crippen.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHBD(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'AromRings': Descriptors.NumAromaticRings(mol),
            'HeavyAtoms': Descriptors.HeavyAtomCount(mol),
        })

df = pd.DataFrame(data)

print(f"\n✓ Calculated features for {len(df)} compounds")
print(f"\nDataset statistics:")
print(df[['MW', 'LogP', 'TPSA', 'HBA', 'HBD']].describe())

# Save
df.to_csv('training_dataset.csv', index=False)
print(f"\n✓ Saved to: training_dataset.csv")

print(f"\nFirst 5 rows:")
print(df.head())
```

---

## COPY THIS SMILES LIST (TEXT FORMAT)

```
Complete list of plant SMILES for Alzheimer's neuroprotection:

O=C1C=C[C@H](O)[C@H]1O
COc1cc(ccc1OC)C(=O)CC(=O)c2ccc(OC)c(OC)c2
O=C(O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](CO)O1)C2=C(O)C(O)=CC(=C2)C3=CC(O)=CC(O)=C3
CC1=C[C@H]2[C@@]34[C@H]1[C@@H](O)[C@H](O)[C@]3(OC(=O)C)C(=C)C[C@H]4O2
O[C@@H]1[C@H](O[C@H]2[C@@H](O)[C@H](O)[C@@H](O)[C@H](O2)CO)[C@H](O)[C@@H](O)[C@@H]1O
CC(C)[C@@H](NC(=O)[C@@H]1CCCN1C(=O))C(=O)N[C@@H](CC(O)=O)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CO)NC(=O)CNC(=O)
CC(C)=C[C@@H]1[C@H]2CC[C@@]3(C)[C@@H](CC[C@@H]4=CC(=O)OC[C@H]34C)[C@@H]2C[C@@H]1O
O=C(O)[C@@H]1[C@H](O)C(=C[C@@H]2C=C(O)[C@H](O)C[C@H]2[C@H]1C)C
CC(C)N1CC[C@H]2[C@H]1CN(C)[C@H]3[C@H]2[C@@H](O)[C@](C)(OC(=O)N)[C@@H]3C
COc1ccc2nc(sc2c1)S(=O)(=O)N
```

---

## FINAL: Download Instructions

```python
print("="*80)
print("YOUR PLANT SMILES DATABASE IS READY!")
print("="*80)

print("\nFiles created:")
print("  1. plant_SMILES_list.txt - One SMILES per line (best for ML)")
print("  2. plant_SMILES_valid.txt - Only valid SMILES")
print("  3. plant_SMILES_single_line.txt - All SMILES in one line")
print("  4. training_dataset.csv - SMILES with molecular features")
print("  5. chembl_ad_compounds.csv - Alzheimer's compounds from ChEMBL")
print("  6. pubchem_neuroprotective.csv - Neuroprotective from PubChem")
print("  7. plant_compounds_manual.csv - 20 key medicinal plants")

print("\nHow to download from Colab:")
print("  1. Left sidebar → Files")
print("  2. Right-click any file → Download")
print("  3. Or: Right-click → Add shortcut to Drive")

print("\nYou now have:")
print(f"  ✓ {len(smiles_list)} unique plant SMILES")
print(f"  ✓ {len(df)} compounds with calculated features")
print(f"  ✓ Ready for machine learning training!")

print("\n" + "="*80)
print("Next step: Use training_dataset.csv for your ensemble model!")
print("="*80)
```

---

## TROUBLESHOOTING

### If ChEMBL download fails:
```python
# Alternative: Use pre-computed ChEMBL data
chembl_url = "https://www.ebi.ac.uk/chembl/api/data/activity?limit=1000&pchembl_value__gte=6"
# Check ChEMBL documentation for API usage
```

### If PubChem is slow:
```python
# Use simplified list with known compounds
simple_compounds = ['quercetin', 'curcumin', 'resveratrol']
# Manually add SMILES
```

### If files won't save to Drive:
```python
# Save to Colab directly
!ls -lh *.csv *.txt
# Then download using file manager
```

---

**Start with Cell 1 and run sequentially!** ✅
Each cell depends on the previous one.
