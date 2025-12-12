import pandas as pd
from chembl_webresource_client.new_client import new_client
import os

def download_target_data(target_name, chembl_id, output_dir="ai_service/data"):
    print(f"Searching for {target_name} ({chembl_id})...")
    
    # Get bioactivities
    activities = new_client.activity
    res = activities.filter(target_chembl_id=chembl_id).filter(standard_type="IC50")

    data = []
    print(f"Fetching IC50 data for {target_name}...")
    for act in res:
        if act['standard_value'] and act['canonical_smiles']:
            data.append({
                'compound_id': act['molecule_chembl_id'],
                'smiles': act['canonical_smiles'],
                'standard_value': float(act['standard_value']),
                'standard_units': act['standard_units'],
                'type': act['standard_type']
            })

    df = pd.DataFrame(data)
    
    # Clean Data
    df = df.dropna(subset=['standard_value', 'smiles'])
    df = df.drop_duplicates(subset=['smiles'])
    
    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"{output_dir}/{target_name}_IC50.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} compounds to {filename}")

if __name__ == "__main__":
    # 1. Alzheimer's: BACE-1 (CHEMBL4822)
    download_target_data("BACE1", "CHEMBL4822")
    
    # 2. Cancer: TP53 (CHEMBL5443 - Check exact ID via search if needed, using general TP53 here or MDM2 usually)
    # Actually, TP53 is a protein, MDM2 is the drug target usually. Let's use MDM2 (CHEMBL3385) for Cancer demo.
    download_target_data("MDM2", "CHEMBL3385")
