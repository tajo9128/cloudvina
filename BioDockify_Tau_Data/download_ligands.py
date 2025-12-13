"""
BioDockify Tau Ligand Generator
Fetches 100 ligands from PubChem and generates 3D SDF files
"""

import os
import time
import requests
from pathlib import Path

# Create directories
os.makedirs("BioDockify_Tau_Data/Ligands", exist_ok=True)

ligand_names = [
    # High Potency / Natural
    "Methylene Blue", "Leuco-methylthioninium", "Emodin", "Daunorubicin", "Doxorubicin", 
    "Oleocanthal", "Epigallocatechin gallate", "Curcumin", "Resveratrol", "Fulvic Acid",
    "Rosmarinic Acid", "Baicalein", "Myricetin", "Gossypetin", "Purpurin",
    "Cyanocobalamin", "Cinnamaldehyde", "Thioflavin S", "Thioflavin T", "Orange G",
    "TRX-0237", "Porphyrin", "Congo Red", "Chrysamine G", "Dopamine",
    "Adrenochrome", "Rifampicin", "Tetracycline", "Doxycycline", "Minocycline",
    "Fisetin", "Quercetin", "Rutin", "Kaempferol", "Luteolin", "Apigenin", "Morin",
    "Silibinin", "Berberine", "Exebryl-1", "Lansoprazole", "Astemizole", "Fluphenazine",
    "Chlorpromazine", "Promethazine", "Azure A", "Azure B", "Toluidine Blue",
    "Shikonin", "Juglone", "Plumbagin", "Tanshinone I", "Tanshinone IIA",
    "Cryptotanshinone", "Salsalate", "Diflunisal", "Aspirin", "Anle138b",
    "Erythrosine", "Rose Bengal", "Phloxine B", "Tartrazine", "Caffeic Acid",
    "Chlorogenic Acid", "Ferulic Acid", "Gallic Acid", "Ellagic Acid", "Punicalagin",
    "Epicatechin", "Catechin", "Spermidine", "Rapamycin", "Lithium Carbonate", "Tideglusib",
    "Tacrine", "Donepezil", "Rivastigmine", "Galantamine", "Memantine", "Huperzine A",
    "Physostigmine", "Scyllo-inositol", "Tramiprosate", "Metformin", "Nilvadipine",
    "Simvastatin", "Lovastatin", "Atorvastatin", "Ibuprofen", "Naproxen", "Indomethacin",
    "Celecoxib", "Diclofenac", "Pioglitazone", "Rosiglitazone", "Nicotine", "Caffeine"
]

print(f"⚗️  Downloading {len(ligand_names)} ligands from PubChem...")
success_count = 0

for idx, name in enumerate(ligand_names, 1):
    try:
        # Clean name for filename
        clean_name = name.replace(" ", "_").replace("-", "_")
        
        # Search PubChem by name to get CID
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
        search_response = requests.get(search_url, timeout=10)
        
        if search_response.status_code == 200:
            cid = search_response.json()['IdentifierList']['CID'][0]
            
            # Download 3D SDF
            sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
            sdf_response = requests.get(sdf_url, timeout=10)
            
            if sdf_response.status_code == 200 and sdf_response.content:
                filename = f"BioDockify_Tau_Data/Ligands/{clean_name}.sdf"
                with open(filename, 'wb') as f:
                    f.write(sdf_response.content)
                print(f"   [{idx}/100] ✅ {name} (CID: {cid})")
                success_count += 1
            else:
                print(f"   [{idx}/100] ⚠️  {name} - No 3D structure")
        else:
            print(f"   [{idx}/100] ❌ {name} - Not found")
        
        time.sleep(0.3)  # Be polite to PubChem API
        
    except Exception as e:
        print(f"   [{idx}/100] ❌ {name} - Error: {str(e)[:50]}")

print(f"\n✅ Downloaded {success_count}/{len(ligand_names)} ligands")
print(f"📁 Location: BioDockify_Tau_Data/Ligands/")
