from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import logging

logger = logging.getLogger(__name__)

class ToxicityService:
    """
    BioDockify NAM Toxicity & ADMET Service (v1.0)
    Implements rule-based and structural alert screening for key safety endpoints.
    Future versions will swap these implementation stubs for trained ML models (ChemBERTa/XGBoost).
    """

    def __init__(self):
        # -- AMES MUTAGENICITY ALERTS (Simplified SMARTS) --
        # Nitro groups, Aromatic amines, Epoxides, Aziridines
        self.mutagenic_smarts = [
            ("[N&D2](=O)[O-]", "Nitro Group"),
            ("c[NH2]", "Aromatic Amine"),
            ("C1OC1", "Epoxide"),
            ("C1NC1", "Aziridine")
        ]
        self.mutagenic_mols = [(Chem.MolFromSmarts(s), name) for s, name in self.mutagenic_smarts]

        # -- hERG INHIBITION ALERTS (Pharmacophore Rules) --
        # Basic rule: High LogP (>3) + Basic Nitrogen (often leads to hERG binding)
        pass

    def predict_comprehensive(self, mol):
        """
        Runs all endpoints and returns a consolidated risk dictionary.
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if not mol:
            return {"error": "Invalid Molecule"}

        results = {
            "hERG": self.assess_herg(mol),
            "ames": self.assess_ames(mol),
            "dili": self.assess_dili(mol),
            "cns_bbb": self.assess_bbb(mol)
        }
        
        # Calculate agreement/confidence
        # (For MVP, confidence is based on how definitive the rule is)
        flags = sum(1 for k, v in results.items() if v.get('risk') == 'HIGH')
        
        results["summary"] = {
            "total_flags": flags,
            "risk_tier": "HIGH" if flags >= 2 else "MEDIUM" if flags == 1 else "LOW"
        }
        return results

    def assess_herg(self, mol):
        """
        hERG Inhibition Risk (Cardiac Toxicity).
        Heuristic: High Lipophilicity (LogP > 3) + Basic Amine + Pi-Stacking features.
        """
        logp = Descriptors.MolLogP(mol)
        has_basic_n = False
        
        # Simple check for aliphatic amine (proxy for basic nitrogen)
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"))
        if matches: has_basic_n = True
        
        risk = "LOW"
        confidence = "MEDIUM"
        reason = []

        if logp > 3.0 and has_basic_n:
            risk = "HIGH"
            reason.append("Lipophylic Basic Amine (hERG Pharmacophore)")
        elif logp > 5.0:
            risk = "MEDIUM"
            reason.append("Very High LogP")

        return {"risk": risk, "probability": 0.85 if risk == "HIGH" else 0.1, "confidence": confidence, "reason": reason}

    def assess_ames(self, mol):
        """
        Ames Mutagenicity (Genotoxicity).
        Heuristic: Structural alerts for DNA-reactive groups.
        """
        risk = "NEGATIVE" # Default safe
        alerts = []
        
        for pat, name in self.mutagenic_mols:
            if mol.HasSubstructMatch(pat):
                alerts.append(name)
        
        if len(alerts) > 0:
            risk = "POSITIVE"
            return {"risk": "HIGH", "prediction": "Mutagenic", "alerts": alerts, "confidence": "HIGH"}
        
        return {"risk": "LOW", "prediction": "Non-Mutagenic", "alerts": [], "confidence": "MEDIUM"}

    def assess_dili(self, mol):
        """
        Drug-Induced Liver Injury (DILI).
        Rule of 2/3 (Chen et al): Daily Dose > 10mg + LogP > 3 => High Risk.
        Since we don't know dose, we flag High LogP + High MW as DILI concern.
        """
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        if logp > 3.0 and mw > 400:
            return {"risk": "MEDIUM", "reason": "High Lipophilicity + MW (DILI Concern)", "confidence": "LOW (Dose Unknown)"}
        
        return {"risk": "LOW", "reason": "Physchem properties largely safe", "confidence": "LOW"}

    def assess_bbb(self, mol):
        """
        Blood-Brain Barrier Penetration.
        Rules: TPSA < 90 and MW < 450 usually penetrate.
        """
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        
        penetrates = (tpsa < 90) and (mw < 450)
        
        prediction = "BBB+" if penetrates else "BBB-"
        return {"prediction": prediction, "properties": {"tpsa": tpsa, "mw": mw}}
