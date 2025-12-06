---
title: "ADMET Prediction and Filtering: Pre-Screening Compounds Before Molecular Docking to Maximize Lead Quality"
description: "Learn how ADMET prediction saves time and improves drug discovery success. Discover practical filtering thresholds for absorption, distribution, metabolism, excretion, and toxicity screening."
keywords: ["ADMET", "drug discovery", "molecular docking", "pharmacokinetics", "toxicity prediction", "Lipinski", "SwissADME", "virtual screening", "drug-likeness", "natural products"]
author: "BioDockify Team"
date: "2024-12-06"
category: "Drug Discovery"
readTime: "9 min read"
---

# ADMET Prediction and Filtering: Pre-Screening Compounds Before Molecular Docking to Maximize Lead Quality

![ADMET Prediction](/blog/images/admet-filtering-hero.jpg)

Finding a compound that binds tightly to your target is only half the battle. **Over 50% of drug candidates fail in clinical trials due to poor pharmacokinetics or toxicity**—not lack of efficacy. By integrating ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) filtering before molecular docking, you can eliminate doomed compounds early, save computational resources, and focus on leads with genuine drug potential.

This practical guide covers ADMET fundamentals, provides ready-to-use filtering thresholds, and shows how pre-docking screening dramatically improves hit quality.

## Why ADMET Must Be Checked Before Docking

### The Brutal Statistics

| Cause of Clinical Failure | Percentage |
|---------------------------|------------|
| Efficacy | 30% |
| **Safety/Toxicity** | **30%** |
| **Pharmacokinetics** | **10%** |
| Commercial viability | 20% |
| Other | 10% |

Nearly **40% of failures** relate to ADMET issues that could be predicted computationally.

### The Cost of Ignoring ADMET

```
Traditional workflow:
Dock 100,000 compounds → Top 1,000 hits → Synthesize 50 → Test in vitro
                                                              ↓
                                          30 fail ADMET screening 😞
                                          
ADMET-first workflow:
100,000 compounds → ADMET filter → 30,000 pass → Dock → Top 500 hits
                                                              ↓
                                     ~400 have acceptable ADMET ✓
```

**Time saved:** Weeks of synthesis/testing on hopeless compounds
**Cost saved:** $10,000-50,000 per failed compound

## Understanding ADMET: The Five Pillars

### A - Absorption

Can the drug get into the body?

| Property | What It Measures | Ideal Values |
|----------|------------------|--------------|
| Oral bioavailability | Drug reaching systemic circulation | > 30% |
| GI absorption | Permeability across intestinal wall | High |
| Caco-2 permeability | In vitro absorption model | > 10-6 cm/s |
| P-gp substrate | Efflux pump liability | No |

**Key factors:** Lipophilicity, molecular weight, hydrogen bonding

### D - Distribution

Where does the drug go in the body?

| Property | What It Measures | Consideration |
|----------|------------------|---------------|
| Volume of distribution (Vd) | Tissue penetration | Target-dependent |
| Plasma protein binding | Free drug available | 70-95% typical |
| BBB penetration | CNS access | Required for neuro drugs |
| Tissue accumulation | Long-term safety | Monitor |

**Critical for:** CNS drugs, cancer drugs requiring tissue penetration

### M - Metabolism

How is the drug broken down?

| Property | What It Measures | Concern |
|----------|------------------|---------|
| CYP450 metabolism | Liver enzyme processing | Drug-drug interactions |
| CYP inhibition | Blocking other drug metabolism | Major safety issue |
| Half-life | Duration of action | Dosing frequency |
| Active metabolites | Metabolite activity | Can be beneficial or harmful |

**Key enzymes:** CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2

### E - Excretion

How does the drug leave the body?

| Property | What It Measures | Importance |
|----------|------------------|------------|
| Renal clearance | Kidney elimination | Dose adjustment in renal impairment |
| Hepatic clearance | Liver elimination | Dose adjustment in liver disease |
| Total clearance | Overall elimination | Determines dosing |

### T - Toxicity

Will the drug cause harm?

| Toxicity Type | Prediction Tools | Critical Threshold |
|---------------|------------------|-------------------|
| **hERG inhibition** | Ion channel models | IC50 > 10 μM |
| **Hepatotoxicity** | Liver injury models | Low risk |
| **Mutagenicity** | AMES test prediction | Negative |
| **Carcinogenicity** | Rodent models | Negative |
| **LD50** | Acute toxicity | Class IV+ preferred |

## In Silico ADMET Tools

### Free Online Tools

| Tool | Strengths | Access |
|------|-----------|--------|
| [SwissADME](http://www.swissadme.ch/) | Comprehensive, visual | Free web |
| [pkCSM](http://biosig.unimelb.edu.au/pkcsm/) | 30+ ADMET endpoints | Free web |
| [ADMETlab 2.0](https://admetmesh.scbdd.com/) | ML-based predictions | Free web |
| [ProTox-II](https://tox-new.charite.de/protox_II/) | Toxicity focused | Free web |
| [PreADMET](https://preadmet.webservice.bmdrc.org/) | Classic calculations | Free web |

### BioDockify Integration

[BioDockify's ADMET panel](/features/admet) provides instant drug-likeness assessment:

```python
# BioDockify API - Get ADMET profile
import requests

def get_admet_profile(smiles, api_key):
    response = requests.post(
        "https://api.biodockify.com/molecules/drug-properties",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"smiles": smiles}
    )
    return response.json()

# Example
profile = get_admet_profile("CC(=O)Oc1ccccc1C(=O)O", api_key)  # Aspirin
print(f"Drug-likeness score: {profile['drug_likeness']['score']}/100")
print(f"Lipinski violations: {profile['lipinski']['violations']}")
```

## Lipinski's Rule of Five

### The Classic Filter

| Rule | Threshold | Rationale |
|------|-----------|-----------|
| MW ≤ 500 | Molecular weight | Absorption limit |
| LogP ≤ 5 | Lipophilicity | Solubility/permeability |
| HBD ≤ 5 | H-bond donors | Membrane permeability |
| HBA ≤ 10 | H-bond acceptors | Membrane permeability |

**Interpretation:** Compounds violating >1 rule have poor oral absorption.

### When to Violate Lipinski

| Situation | Example | Rationale |
|-----------|---------|-----------|
| Non-oral routes | Injectable drugs | Bypass GI absorption |
| Active transport | Sugars, amino acids | Carrier-mediated |
| Natural products | Cyclosporin (MW 1203) | Evolution-optimized |
| Biologics | Peptides, antibodies | Different rules apply |

**Important:** ~6% of approved oral drugs violate Lipinski, especially natural products.

## Practical Filtering Thresholds

### Ready-to-Use Criteria for Virtual Screening

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def admet_filter(smiles, strict=True):
    """
    Apply practical ADMET filters for virtual screening.
    Returns True if compound passes, False otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    # Calculate properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    
    if strict:
        # Strict criteria for oral drugs
        criteria = {
            'mw': 150 <= mw <= 500,
            'logp': -0.5 <= logp <= 5.0,
            'hbd': hbd <= 5,
            'hba': hba <= 10,
            'tpsa': 20 <= tpsa <= 130,
            'rot_bonds': rot_bonds <= 10
        }
    else:
        # Relaxed criteria (natural products, non-oral)
        criteria = {
            'mw': 100 <= mw <= 800,
            'logp': -2 <= logp <= 7,
            'hbd': hbd <= 7,
            'hba': hba <= 15,
            'tpsa': tpsa <= 180,
            'rot_bonds': rot_bonds <= 15
        }
    
    return all(criteria.values())

# Usage
smiles_list = ["CCO", "CC(=O)NC1=CC=C(O)C=C1", "C1=CC=CC=C1"]  # Example
passed = [smi for smi in smiles_list if admet_filter(smi)]
print(f"{len(passed)}/{len(smiles_list)} compounds passed ADMET filter")
```

### Quick Reference Table

| Property | Optimal | Acceptable | Flag |
|----------|---------|------------|------|
| MW | 200-400 | 150-500 | >500 |
| LogP | 1-3 | -0.5 to 5 | >5 |
| HBD | 0-2 | 0-5 | >5 |
| HBA | 2-6 | 0-10 | >10 |
| TPSA | 40-90 | 20-130 | >140 |
| RotBonds | 2-5 | 0-10 | >10 |
| **Lipinski violations** | **0** | **0-1** | **>1** |

## Special Considerations for Natural Products

### The Natural Product Challenge

Phytochemicals often violate "drug-like" rules:

| Property | Synthetic Drugs | Natural Products |
|----------|-----------------|------------------|
| MW | ~350 | ~450+ |
| LogP | ~2.5 | Highly variable |
| Stereochemistry | Simple | Complex |
| Glycosylation | Rare | Common |

### Adjusted Thresholds for Phytochemicals

```python
def natural_product_filter(smiles):
    """Relaxed ADMET filter for phytochemicals."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    
    # Natural product-appropriate thresholds
    criteria = {
        'mw': mw <= 800,      # Extended for glycosides
        'logp': logp <= 8,    # Allow more lipophilic terpenoids
        'hbd': hbd <= 8,      # Polyhydroxylated compounds
        'hba': hba <= 15,     # Sugar moieties
    }
    
    # Count violations (allow up to 2 for natural products)
    violations = sum(1 for passed in criteria.values() if not passed)
    
    return violations <= 2
```

### Why Natural Products Still Work

1. **Active transport mechanisms** bypass passive diffusion
2. **Cell membrane integration** for lipophilic structures
3. **Prodrug effects** from glycosylation
4. **Evolution-tested** for biological activity

## Integrated Workflow: ADMET → Docking → Lead

### The Optimal Pipeline

```
                    Full Library (100,000)
                           │
                  ┌────────▼────────┐
                  │  ADMET Filters   │
                  │  SwissADME/      │
                  │  BioDockify      │
                  └────────┬────────┘
                           │
                   Pass ADMET (~40,000)
                           │
                  ┌────────▼────────┐
                  │    Substructure  │
                  │    PAINS Filter  │
                  └────────┬────────┘
                           │
                   PAINS-free (~35,000)
                           │
                  ┌────────▼────────┐
                  │ BioDockify       │
                  │ Molecular Docking│
                  └────────┬────────┘
                           │
                   Top 1% (~350 hits)
                           │
                  ┌────────▼────────┐
                  │ Visual Inspection│
                  │ Interaction check│
                  └────────┬────────┘
                           │
                   Priority leads (~50)
                           │
                  ┌────────▼────────┐
                  │ Synthesis/Assay  │
                  └─────────────────┘
```

### Expected Improvements

| Metric | Without ADMET Filter | With ADMET Filter |
|--------|---------------------|-------------------|
| Compounds docked | 100,000 | 35,000 |
| Docking time | 48 hours | 17 hours |
| Hits with good ADMET | ~30% | ~85% |
| Compounds synthesized | 50 | 50 |
| Leads progressing | 15 | 40 |

## Integration with BioDockify

### The Complete Workflow

```python
# Step 1: Upload library to BioDockify
library = upload_compound_library("compounds.sdf")

# Step 2: Apply built-in ADMET filter
filtered = apply_admet_filter(
    library,
    lipinski_violations_max=1,
    logp_max=5,
    mw_max=500,
    pains_filter=True
)

# Step 3: Run docking on filtered set
job = submit_docking_job(
    receptor="target.pdbqt",
    ligands=filtered,
    exhaustiveness=16
)

# Step 4: Get results with drug-likeness scores
results = get_job_results(job.id)
for hit in results.top_hits:
    print(f"{hit.name}: Score={hit.docking_score}, "
          f"Drug-likeness={hit.drug_likeness_score}")
```

### BioDockify ADMET Features

- **[Drug-likeness scoring](/features/admet)** with Lipinski/Veber checks
- **PAINS filter** to remove interference compounds
- **External tool links** to SwissADME, ProTox-II, pkCSM
- **One-click filtering** during library upload

## Practical Example: Screening Natural Products

### Scenario

Screening 10,000 phytochemicals against a CNS target:

```python
# Load natural product library
np_library = load_library("phytochemicals_10k.sdf")

# Apply natural product-appropriate filters
filtered = []
for mol in np_library:
    smi = Chem.MolToSmiles(mol)
    
    # Calculate key properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Natural product criteria + CNS requirement
    if (mw <= 600 and 
        logp <= 5 and 
        tpsa <= 90):  # CNS penetration
        filtered.append(mol)

print(f"CNS-penetrant subset: {len(filtered)} compounds")
# Result: ~3,500 compounds ready for docking
```

## Conclusion

ADMET filtering is not optional—it's essential for efficient drug discovery. By eliminating compounds with poor pharmacokinetics or toxicity **before** expensive docking and synthesis, you:

1. **Save time** by reducing docking workload by 50-70%
2. **Save money** by not synthesizing doomed compounds
3. **Improve hit quality** with 2-3× more progressing to leads
4. **Strengthen publications** with clinically relevant candidates

The modern workflow is simple: **Filter first, dock second, synthesize smart.**

**Start your ADMET-aware drug discovery with [BioDockify](https://biodockify.com/signup)** — integrated filtering and docking in one platform.

---

## Related Articles

- [Virtual Screening for Natural Product Drug Discovery](/blog/virtual-screening-natural-products)
- [Drug-Likeness Calculator in BioDockify](/features/admet)
- [From Protein Structure to Drug Binding](/blog/protein-preparation-guide)

## External Resources

- [SwissADME Web Server](http://www.swissadme.ch/)
- [pkCSM Pharmacokinetics](http://biosig.unimelb.edu.au/pkcsm/)
- [ProTox-II Toxicity Prediction](https://tox-new.charite.de/protox_II/)
- [ADMETlab 2.0](https://admetmesh.scbdd.com/)
