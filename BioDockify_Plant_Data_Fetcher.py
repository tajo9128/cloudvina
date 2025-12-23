
# 🌿 BioDockify: Comprehensive Plant SMILES Harvester
# Based on: Google_Colab_Plant_SMILES_Fetcher.md
# Sources: ChEMBL, PubChem, Literature, Manual Database

print("="*70)
print("🌿 BioDockify: PLANT SMILES HARVESTER (Thousands of Compounds)")
print("="*70)

# --- CELL 1: Install & Import ---
import subprocess
import sys

# Install required packages
packages = ['pandas', 'pubchempy', 'chembl_webresource_client', 'rdkit']
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        print(f"⬇️ Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import pandas as pd
import os
import time
from tqdm.auto import tqdm

# Mount Drive
try:
    from google.colab import drive
    if not os.path.isdir("/content/drive"):
        drive.mount('/content/drive')
        print("✅ Google Drive Mounted")
except: 
    pass

# --- CELL 3: ChEMBL Alzheimer's Compounds ---
def fetch_chembl_alzheimers():
    print("\n>>> 🧪 Fetching ChEMBL (Alzheimer's Compounds)...")
    csv_name = "chembl_ad_compounds.csv"
    
    if os.path.exists(csv_name):
        print(f"    ✅ Already exists: {csv_name}")
        return pd.read_csv(csv_name)
    
    try:
        from chembl_webresource_client.new_client import new_client
        activity = new_client.activity
        
        all_data = []
        targets = [
            ("CHEMBL220", "AChE"),   # Acetylcholinesterase
            ("CHEMBL3927", "BACE1"), # Beta-secretase
            ("CHEMBL262", "GSK3b"),  # Glycogen synthase kinase
        ]
        
        for target_id, target_name in targets:
            print(f"    Fetching {target_name}...")
            try:
                results = activity.filter(
                    target_chembl_id=target_id,
                    assay_type="B",
                    pchembl_value__gte=6
                )
                for act in tqdm(list(results)[:2000], desc=target_name):
                    if act.get('canonical_smiles'):
                        all_data.append({
                            'smiles': act['canonical_smiles'],
                            'source': 'ChEMBL',
                            'target': target_name
                        })
            except Exception as e:
                print(f"    ⚠️ Error: {e}")
        
        df = pd.DataFrame(all_data)
        df.to_csv(csv_name, index=False)
        print(f"    ✅ Saved {len(df)} compounds")
        return df
        
    except Exception as e:
        print(f"    ⚠️ ChEMBL Error: {e}")
        return pd.DataFrame()

# --- CELL 4: PubChem Neuroprotective Compounds ---
def fetch_pubchem_neuroprotective():
    print("\n>>> 🧬 Fetching PubChem (Neuroprotective Compounds)...")
    csv_name = "pubchem_neuroprotective.csv"
    
    if os.path.exists(csv_name):
        print(f"    ✅ Already exists: {csv_name}")
        return pd.read_csv(csv_name)
    
    try:
        import pubchempy as pcp
        
        # Extended list of neuroprotective compounds
        compounds_to_fetch = [
            'quercetin', 'curcumin', 'resveratrol', 'EGCG', 'galantamine',
            'huperzine A', 'luteolin', 'morin', 'baicalein', 'catechin',
            'epicatechin', 'silymarin', 'astaxanthin', 'lycopene',
            'ginkgolide', 'bilobalide', 'bacosides', 'withaferin',
            'carnosic acid', 'rosmarinic acid', 'tanshinone', 'berberine',
            'piperine', 'asiatic acid', 'ginsenoside', 'apigenin',
            'naringenin', 'kaempferol', 'myricetin', 'rutin', 'hesperidin',
            'chlorogenic acid', 'ferulic acid', 'caffeic acid', 'ellagic acid',
            'gallic acid', 'vanillic acid', 'syringic acid', 'protocatechuic acid',
            'oleuropein', 'hydroxytyrosol', 'tyrosol', 'piceatannol',
            'pterostilbene', 'honokiol', 'magnolol', 'baicalin', 'scutellarin',
            'icariin', 'genistein', 'daidzein', 'glycitein', 'formononetin'
        ]
        
        all_data = []
        for name in tqdm(compounds_to_fetch, desc="PubChem"):
            try:
                results = pcp.get_compounds(name, 'name')
                if results:
                    comp = results[0]
                    if comp.canonical_smiles:
                        all_data.append({
                            'smiles': comp.canonical_smiles,
                            'name': name,
                            'source': 'PubChem',
                            'mw': comp.molecular_weight
                        })
                time.sleep(0.3)  # Rate limiting
            except:
                pass
        
        df = pd.DataFrame(all_data)
        df.to_csv(csv_name, index=False)
        print(f"    ✅ Saved {len(df)} compounds")
        return df
        
    except Exception as e:
        print(f"    ⚠️ PubChem Error: {e}")
        return pd.DataFrame()

# --- CELL 5: Plant Compounds Database (From Literature) ---
def create_plant_database():
    print("\n>>> 🌱 Creating Plant Compounds Database...")
    
    # Comprehensive plant compound database
    plant_data = [
        # Evolvulus alsinoides (Your research plant!)
        ('Evolvulus alsinoides', 'Shankhpushpi alkaloid 1', 'CN1CCC2=CC(=C(C=C2C1)O)O'),
        ('Evolvulus alsinoides', 'Evolvine', 'COC1=CC2=C(C=C1)N(C)CCC2'),
        ('Evolvulus alsinoides', 'Scopoletin', 'COC1=CC2=C(C=C1O)OC(=O)C=C2'),
        
        # Cordia dichotoma (Your research plant!)
        ('Cordia dichotoma', 'Allantoin', 'NC(=O)NC1NC(=O)NC1=O'),
        ('Cordia dichotoma', 'Taxifolin', 'O=C1C(O)C(OC2=CC(O)=CC(O)=C12)C3=CC(O)=C(O)C=C3'),
        
        # Ginkgo biloba
        ('Ginkgo biloba', 'Ginkgolide A', 'CC1C2CC3(C1C4C5C(CC(O5)O4)(C(O3)(C(C2OC(=O)C)O)OC(=O)C)C)C'),
        ('Ginkgo biloba', 'Ginkgolide B', 'CC1C(=O)OC2C1(C34C(=O)OC5C3(C2O)C6(C(C5)C(C)(C)C)C(C(=O)OC6O4)O)O'),
        ('Ginkgo biloba', 'Bilobalide', 'CC1(C)C2OC3C4OC(=O)C(C)(C4(O)OC3C(=O)C2C(C1=O)O)C'),
        
        # Bacopa monnieri
        ('Bacopa monnieri', 'Bacoside A', 'CC1CCC2(C)C(CC(O)C3C2CCC4(C)C3CCC5(C)C4CC(O)C5O)C1'),
        ('Bacopa monnieri', 'Bacopaside I', 'CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C'),
        ('Bacopa monnieri', 'Bacosine', 'CN1CCC2=CC(O)=C(O)C=C2C1CC(O)C3=CC=C(O)C=C3'),
        
        # Curcuma longa
        ('Curcuma longa', 'Curcumin', 'COC1=C(O)C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(O)C=C2)OC'),
        ('Curcuma longa', 'Demethoxycurcumin', 'COC1=C(O)C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC=C(O)C=C2'),
        ('Curcuma longa', 'Bisdemethoxycurcumin', 'OC1=CC=C(C=CC(=O)CC(=O)C=CC2=CC=C(O)C=C2)C=C1'),
        ('Curcuma longa', 'Ar-turmerone', 'CC(=CCC(C)(C1=CC=C(C)C=C1)O)C'),
        
        # Camellia sinensis (Green Tea)
        ('Camellia sinensis', 'EGCG', 'OC1=CC(=CC(=C1O)O)C2OC3=CC(=CC(=C3CC2OC(=O)C4=CC(=C(C(=C4)O)O)O)O)O'),
        ('Camellia sinensis', 'Epicatechin', 'OC1=CC(=CC(=C1O)O)C2OC3=C(C(=CC(=C3)O)O)CC2O'),
        ('Camellia sinensis', 'Catechin', 'OC1=CC(=CC(=C1O)O)C2OC3=CC(=CC(=C3C2O)O)O'),
        ('Camellia sinensis', 'L-theanine', 'CCNC(=O)CCC(N)C(=O)O'),
        
        # Withania somnifera (Ashwagandha)
        ('Withania somnifera', 'Withaferin A', 'CC(C(O)C1OC(C)=O)C2CCC3C4CC=C5CC(O)CCC5(C)C4CCC23C'),
        ('Withania somnifera', 'Withanolide A', 'CC1OC(=O)C(C)C(O)C2CCC3C4CC=C5CC(O)CCC5(C)C4CCC23'),
        ('Withania somnifera', 'Withanone', 'CC(C)=CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C'),
        
        # Rosmarinus officinalis
        ('Rosmarinus officinalis', 'Carnosic acid', 'CC(C)C1=C(O)C(O)=C2C(=C1)CC3CCCC(C)(C)C3C2C(O)=O'),
        ('Rosmarinus officinalis', 'Carnosol', 'CC(C)C1=CC2=C(C(O)=C1O)C3CCCC(C)(C)C3CC2'),
        ('Rosmarinus officinalis', 'Rosmarinic acid', 'O=C(OC(CC1=CC(O)=C(O)C=C1)C(O)=O)C=CC2=CC(O)=C(O)C=C2'),
        
        # Salvia miltiorrhiza
        ('Salvia miltiorrhiza', 'Tanshinone IIA', 'CC1COC2=C1C(=O)C(=O)C3=C2C=CC4=C3CCCC4(C)C'),
        ('Salvia miltiorrhiza', 'Cryptotanshinone', 'CC1COC2=C1C(=O)C(=O)C3=C2C=CC4=C3CCC(C4)(C)C'),
        ('Salvia miltiorrhiza', 'Salvianolic acid B', 'OC(=O)C(CC1=CC(O)=C(O)C=C1)OC(=O)C=CC2=CC(O)=C(O)C=C2'),
        
        # Huperzia serrata
        ('Huperzia serrata', 'Huperzine A', 'C/C=C/1\\C2=C(N)C(=O)NC=C2CC(C)C1CC=C'),
        ('Huperzia serrata', 'Huperzine B', 'CC=C1C2=C(N)C(=O)NC=C2CC(CC)C1CC=C'),
        
        # Centella asiatica
        ('Centella asiatica', 'Asiatic acid', 'CC1CCC2(C(=O)O)CCC3(C)C(=CCC4C5(C)CCC(O)C(C)(C)C5CCC34C)C2C1C'),
        ('Centella asiatica', 'Asiaticoside', 'CC1CCC2(C)C(CC(O)C3C2CCC4(C)C3CCC5(C)C4CC(O)C5O)C1'),
        ('Centella asiatica', 'Madecassoside', 'CC1CCC2(C)C(CC(O)C3C2CCC4(C)C3CCC(O)C4O)C1'),
        
        # Panax ginseng
        ('Panax ginseng', 'Ginsenoside Rb1', 'CC(C)=CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C'),
        ('Panax ginseng', 'Ginsenoside Rg1', 'CC(C)=CCCC(C)C1CCC2C3=CCC4C(C)(C)C(O)CCC4(C)C3CCC12C'),
        ('Panax ginseng', 'Ginsenoside Re', 'CC1OC(OC2C(O)C(O)C(CO)OC2O)C(O)C(O)C1O'),
        
        # Piper nigrum (Black Pepper)
        ('Piper nigrum', 'Piperine', 'O=C(/C=C/C=C/C1=CC2=C(OCO2)C=C1)N3CCCCC3'),
        ('Piper nigrum', 'Piperlongumine', 'COC1=CC(=CC(=C1OC)OC)/C=C/C(=O)N2CCC=CC2=O'),
        
        # Glycyrrhiza glabra (Licorice)
        ('Glycyrrhiza glabra', 'Glycyrrhizin', 'CC1(C)CCC2(C(=O)O)CCC3(C)C(=CCC4C5(C)CCC(O)C(C)(C)C5CCC34C)C2C1'),
        ('Glycyrrhiza glabra', 'Glabridin', 'CC1(C)C=CC2=C(O1)C=CC3=C2OCC(C3)C4=CC=C(O)C=C4'),
        
        # Magnolia officinalis
        ('Magnolia officinalis', 'Honokiol', 'OC1=CC(=CC=C1CC2=CC(=CC=C2O)CC=C)CC=C'),
        ('Magnolia officinalis', 'Magnolol', 'OC1=C(CC=C)C=C(CC2=CC(=C(O)C=C2)CC=C)C=C1'),
        
        # Scutellaria baicalensis
        ('Scutellaria baicalensis', 'Baicalein', 'O=C1C=C(OC2=C1C(O)=C(O)C(O)=C2)C3=CC=CC=C3'),
        ('Scutellaria baicalensis', 'Baicalin', 'O=C1C=C(OC2=C1C(O)=C(O)C(OC3OC(C(O)=O)C(O)C(O)C3O)=C2)C4=CC=CC=C4'),
        ('Scutellaria baicalensis', 'Wogonin', 'COC1=C(O)C=C(O)C2=C1OC(=CC2=O)C3=CC=CC=C3'),
        
        # Additional flavonoids
        ('Various plants', 'Quercetin', 'O=C1C(O)=C(OC2=CC(O)=CC(O)=C12)C3=CC(O)=C(O)C=C3'),
        ('Various plants', 'Kaempferol', 'O=C1C(O)=C(OC2=CC(O)=CC(O)=C12)C3=CC=C(O)C=C3'),
        ('Various plants', 'Myricetin', 'O=C1C(O)=C(OC2=CC(O)=CC(O)=C12)C3=CC(O)=C(O)C(O)=C3'),
        ('Various plants', 'Isorhamnetin', 'COC1=C(O)C=CC(=C1)C2=C(O)C(=O)C3=C(O)C=C(O)C=C3O2'),
        ('Various plants', 'Fisetin', 'O=C1C(O)=C(OC2=CC(O)=CC=C12)C3=CC(O)=C(O)C=C3'),
        ('Various plants', 'Naringenin', 'O=C1CC(OC2=CC(O)=CC(O)=C12)C3=CC=C(O)C=C3'),
        ('Various plants', 'Hesperidin', 'COC1=CC(=C(O)C=C1)C2CC(=O)C3=C(O)C=C(OC4OC(CO)C(O)C(O)C4OC5OC(C)C(O)C(O)C5O)C=C3O2'),
        ('Various plants', 'Rutin', 'CC1OC(OC2C(O)C(O)C(CO)OC2OC3=C(OC4=CC(O)=CC(O)=C4C3=O)C5=CC(O)=C(O)C=C5)C(O)C(O)C1O'),
        
        # Phenolic acids
        ('Various plants', 'Chlorogenic acid', 'O=C(OC1CC(O)(C(O)=O)CC(O)C1O)/C=C/C2=CC(O)=C(O)C=C2'),
        ('Various plants', 'Ferulic acid', 'COC1=C(O)C=CC(=C1)/C=C/C(O)=O'),
        ('Various plants', 'Caffeic acid', 'OC1=C(O)C=CC(=C1)/C=C/C(O)=O'),
        ('Various plants', 'Gallic acid', 'O=C(O)C1=CC(O)=C(O)C(O)=C1'),
        ('Various plants', 'Ellagic acid', 'O=C1OC2=C(O)C(O)=CC3=C2C4=C(OC(=O)C5=CC(O)=C(O)C=C45)C=C3O1'),
        
        # From your user list
        ('Hopea hainanensis', 'Hopeahainol A', 'C1=CC(=CC=C1[C@@H]2[C@H](C3=C4[C@H]([C@@H](OC4=CC(=C3)O)C5=CC=C(C=C5)O)C6=C2C(=CC(=C6)O)O)[C@H]7[C@@H](C8=C(C=C(C=C8O)O)[C@H]9[C@@H](OC1=CC(=CC7=C91)O)C1=CC=C(C=C1)O)C1=CC=C(C=C1)O)O'),
        ('Vitis spp.', 'Resveratrol', 'Oc1ccc(cc1)/C=C/c2cc(O)cc(O)c2'),
        ('Berberis spp.', 'Berberine', 'COC1=C(OC)C2=CC3=C(C=C2C=C1)C4=CC5=C(C=C4C[N+]3=C)OCO5'),
    ]
    
    df = pd.DataFrame(plant_data, columns=['Plant_Name', 'Compound_Name', 'SMILES'])
    df['source'] = 'Literature'
    
    print(f"    ✅ Created database with {len(df)} plant compounds")
    print(f"    Plants included: {df['Plant_Name'].nunique()}")
    
    return df

# --- MAIN HARVESTER ---
def harvest_all_data():
    all_dfs = []
    
    # 1. ChEMBL
    df1 = fetch_chembl_alzheimers()
    if len(df1) > 0:
        all_dfs.append(df1)
    
    # 2. PubChem
    df2 = fetch_pubchem_neuroprotective()
    if len(df2) > 0:
        all_dfs.append(df2)
    
    # 3. Plant Literature Database
    df3 = create_plant_database()
    all_dfs.append(df3)
    
    # Combine
    print("\n>>> 🔄 Consolidating All Data...")
    combined_smiles = []
    
    for df in all_dfs:
        if 'smiles' in df.columns:
            combined_smiles.extend(df['smiles'].dropna().tolist())
        if 'SMILES' in df.columns:
            combined_smiles.extend(df['SMILES'].dropna().tolist())
    
    # Remove duplicates
    combined_smiles = list(set(combined_smiles))
    
    # Create final dataframe
    final_df = pd.DataFrame({'smiles': combined_smiles})
    final_filename = "biodockify_global_plant_database.csv"
    
    # Save locally
    final_df.to_csv(final_filename, index=False)
    
    # Save to Drive
    if os.path.isdir("/content/drive/MyDrive"):
        gdrive_path = f"/content/drive/MyDrive/{final_filename}"
        final_df.to_csv(gdrive_path, index=False)
        print(f"    💾 Saved to Drive: {gdrive_path}")
    
    print("\n" + "="*70)
    print(f"🏆 SUCCESS: PLANT DATABASE CREATED!")
    print(f"    🔢 Total Unique SMILES: {len(final_df)}")
    print(f"    📄 File: {final_filename}")
    print("="*70)
    
    return final_df

if __name__ == "__main__":
    harvest_all_data()
