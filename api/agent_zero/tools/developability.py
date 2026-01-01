try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def assess_developability(smiles: str) -> dict:
    """
    Analyzes a molecule for developability risks (Toxicity, Solubility, Stability).
    Uses RDKit to calculate properties and check for structural alerts.
    
    Args:
        smiles (str): The chemical structure in SMILES format.
    
    Returns:
        dict: Assessment report with flags and calculated properties.
    """
    if not RDKIT_AVAILABLE:
        return {"error": "RDKit is not installed on the server. Cannot perform analysis."}

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"error": "Invalid SMILES string."}

    # 1. Properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)

    # 2. Risk Flags (Lipinski Rule of 5 violations)
    violations = []
    if mw > 500: violations.append("MW > 500")
    if logp > 5: violations.append("LogP > 5 (Likely Insoluble)")
    if hbd > 5: violations.append("H-Bond Donors > 5")
    if hba > 10: violations.append("H-Bond Acceptors > 10")

    # 3. Structural Alerts (Simplified PAINS / Reactivity)
    # Define some SMARTS patterns for common issues
    alerts = []
    
    # Nitro group (often mutagenic/toxic)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")):
        alerts.append("Nitro group (Toxicity Risk)")
        
    # Aldehyde (Reactive)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3H1](=O)[#6]")):
        alerts.append("Aldehyde (High Reactivity)")
        
    # Michael Acceptor (C=C-C=O) - covalent binding risk
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C,c]=[C,c]-[C,c]=O")):
        alerts.append("Michael Acceptor (Potential Covalent Binder)")

    # 4. Overall Assessment
    risk_level = "Low"
    if len(violations) >= 2 or len(alerts) >= 1:
        risk_level = "Medium"
    if len(violations) >= 3 or len(alerts) >= 2:
        risk_level = "High"

    return {
        "smiles": smiles,
        "properties": {
            "mw": round(mw, 2),
            "logp": round(logp, 2),
            "hbd": hbd,
            "hba": hba,
            "tpsa": round(tpsa, 2),
            "rotatable_bonds": rotatable_bonds
        },
        "lipinski_violations": violations,
        "structural_alerts": alerts,
        "risk_level": risk_level,
        "conclusion": "Suitable for development" if risk_level == "Low" else "Optimization required"
    }

# Fallback for simple testing if needed
if __name__ == "__main__":
    print(assess_developability("CC(=O)OC1=CC=CC=C1C(=O)O")) # Aspirin
