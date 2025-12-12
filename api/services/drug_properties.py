"""
Drug Properties Service
Calculates drug-likeness, ADMET predictions, and molecular properties using RDKit.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, FilterCatalog, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DrugPropertiesCalculator:
    """
    Calculates drug-like properties and filters for small molecules.
    Uses RDKit for all calculations - 100% free and open source.
    """
    
    def __init__(self):
        # Initialize PAINS filter
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains_catalog = FilterCatalog.FilterCatalog(params)
    
    def calculate_all(self, smiles: str) -> Dict:
        """
        Calculate all drug properties for a given SMILES string.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary with all calculated properties
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES string"}
        
        try:
            return {
                "smiles": smiles,
                "molecular_properties": self._get_molecular_properties(mol),
                "lipinski": self._check_lipinski(mol),
                "veber": self._check_veber(mol),
                "drug_likeness": self._assess_drug_likeness(mol),
                "pains": self._check_pains(mol),
                "lead_likeness": self._check_lead_likeness(mol),
                "bbb": self._predict_bbb_permeability(mol),
                "toxicity": self._check_toxicity_alerts(mol),
                "admet_links": self._get_admet_links(smiles),
                "summary": self._get_summary(mol)
            }
        except Exception as e:
            logger.error(f"Error calculating properties: {e}")
            return {"error": str(e)}
    
    def _get_molecular_properties(self, mol) -> Dict:
        """Calculate basic molecular properties."""
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "aromatic_rings": Descriptors.NumAromaticRings(mol),
            "heavy_atoms": Lipinski.HeavyAtomCount(mol),
            "fraction_sp3": round(Descriptors.FractionCSP3(mol), 2),
            "molar_refractivity": round(Descriptors.MolMR(mol), 2),
            "num_rings": Descriptors.RingCount(mol)
        }
    
    def _check_lipinski(self, mol) -> Dict:
        """
        Check Lipinski's Rule of 5 for oral bioavailability.
        A drug-like compound should have:
        - MW ≤ 500
        - LogP ≤ 5
        - HBD ≤ 5
        - HBA ≤ 10
        """
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        violations = []
        if mw > 500:
            violations.append(f"MW ({mw:.1f}) > 500")
        if logp > 5:
            violations.append(f"LogP ({logp:.2f}) > 5")
        if hbd > 5:
            violations.append(f"HBD ({hbd}) > 5")
        if hba > 10:
            violations.append(f"HBA ({hba}) > 10")
        
        return {
            "passed": len(violations) == 0,
            "violations": len(violations),
            "violation_details": violations,
            "rules": {
                "mw_ok": mw <= 500,
                "logp_ok": logp <= 5,
                "hbd_ok": hbd <= 5,
                "hba_ok": hba <= 10
            }
        }
    
    def _check_veber(self, mol) -> Dict:
        """
        Check Veber rules for oral bioavailability.
        - Rotatable bonds ≤ 10
        - TPSA ≤ 140 Å²
        """
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        
        violations = []
        if rot_bonds > 10:
            violations.append(f"Rotatable bonds ({rot_bonds}) > 10")
        if tpsa > 140:
            violations.append(f"TPSA ({tpsa:.1f}) > 140")
        
        return {
            "passed": len(violations) == 0,
            "violations": len(violations),
            "violation_details": violations,
            "rules": {
                "rotatable_bonds_ok": rot_bonds <= 10,
                "tpsa_ok": tpsa <= 140
            }
        }
    
    def _check_lead_likeness(self, mol) -> Dict:
        """
        Check lead-likeness criteria (more restrictive than Lipinski).
        - MW 250-350
        - LogP ≤ 3.5
        - Rotatable bonds ≤ 7
        """
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        
        violations = []
        if mw < 250 or mw > 350:
            violations.append(f"MW ({mw:.1f}) not in 250-350 range")
        if logp > 3.5:
            violations.append(f"LogP ({logp:.2f}) > 3.5")
        if rot_bonds > 7:
            violations.append(f"Rotatable bonds ({rot_bonds}) > 7")
        
        return {
            "passed": len(violations) == 0,
            "violations": len(violations),
            "violation_details": violations
        }
    
    def _check_pains(self, mol) -> Dict:
        """
        Check for PAINS (Pan-Assay Interference Compounds).
        These are compounds that give false positives in assays.
        """
        matches = self.pains_catalog.GetMatches(mol)
        pains_alerts = [match.GetDescription() for match in matches]
        
        return {
            "passed": len(pains_alerts) == 0,
            "num_alerts": len(pains_alerts),
            "alerts": pains_alerts[:5]  # Limit to first 5
        }
    
    def _assess_drug_likeness(self, mol) -> Dict:
        """
        Overall drug-likeness assessment combining multiple rules.
        """
        lipinski = self._check_lipinski(mol)
        veber = self._check_veber(mol)
        pains = self._check_pains(mol)
        
        # Calculate a simple drug-likeness score (0-100)
        score = 100
        score -= lipinski["violations"] * 15
        score -= veber["violations"] * 10
        score -= pains["num_alerts"] * 20
        score = max(0, min(100, score))
        
        # Determine category
        if score >= 80 and lipinski["passed"] and pains["passed"]:
            category = "Drug-like"
            color = "green"
        elif score >= 50:
            category = "Moderate"
            color = "yellow"
        else:
            category = "Poor"
            color = "red"
        
        return {
            "score": score,
            "category": category,
            "color": color,
            "recommendation": self._get_recommendation(lipinski, veber, pains)
        }
    
    def _get_recommendation(self, lipinski, veber, pains) -> str:
        """Generate recommendation based on violations."""
        issues = []
        
        if not lipinski["passed"]:
            issues.append("Consider reducing molecular weight or lipophilicity")
        if not veber["passed"]:
            issues.append("Consider reducing flexibility (rotatable bonds) or polarity (TPSA)")
        if not pains["passed"]:
            issues.append("Contains PAINS alerts - may give false positives in assays")
        
        if not issues:
            return "Good drug-like properties. Suitable for further development."
        
        return " | ".join(issues)
    
    def _get_admet_links(self, smiles: str) -> Dict:
        """
        Generate links to external ADMET prediction tools.
        All these tools are FREE to use.
        """
        import urllib.parse
        encoded_smiles = urllib.parse.quote(smiles, safe='')
        
        return {
            "swissadme": {
                "name": "SwissADME",
                "url": f"http://www.swissadme.ch/index.php?smiles={encoded_smiles}",
                "description": "Pharmacokinetics, drug-likeness, medicinal chemistry"
            },
            "protox": {
                "name": "ProTox-II",
                "url": "https://tox-new.charite.de/protox_II/",
                "description": "Toxicity prediction (LD50, hepatotoxicity, carcinogenicity)"
            },
            "admetlab": {
                "name": "ADMETlab 2.0",
                "url": f"https://admetmesh.scbdd.com/service/evaluation/index",
                "description": "Comprehensive ADMET prediction"
            },
            "pkcsm": {
                "name": "pkCSM",
                "url": "https://biosig.lab.uq.edu.au/pkcsm/prediction",
                "description": "Pharmacokinetic properties"
            },
            "swisstarget": {
                "name": "SwissTargetPrediction", 
                "url": f"http://www.swisstargetprediction.ch/result.php?smiles={encoded_smiles}",
                "description": "Predict protein targets"
            }
        }
    
    def _predict_bbb_permeability(self, mol) -> Dict:
        """
        Predict Blood-Brain Barrier (BBB) permeability using physicochemical rules.
        """
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        hbd = Lipinski.NumHDonors(mol)
        
        # Rule: TPSA < 90 and MW < 450 usually indicates good CNS penetration
        is_permeable = (tpsa < 90) and (mw < 450) and (hbd < 3)
        
        return {
            "permeable": is_permeable,
            "rules": {
                "tpsa_ok": tpsa < 90,
                "mw_ok": mw < 450,
                "hbd_ok": hbd < 3
            },
            "description": "Likely to cross BBB" if is_permeable else "Poor CNS penetration"
        }

    def _check_toxicity_alerts(self, mol) -> Dict:
        """
        Check for common structural alerts (Toxicophores).
        """
        alerts = {
            "Nitro Group": "[N+](=O)[O-]",
            "Hydrazine": "[NX3][NX3]",
            "Michael Acceptor": "[CX3]=[CX3]-[CX3](=[OX1])", # Basic pattern
            "Alkyl Halide": "[CX4][F,Cl,Br,I]",
            "Aldehyde": "[CX3H1](=O)[#6]" 
        }
        
        found_alerts = []
        for name, smarts in alerts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                found_alerts.append(name)
        
        return {
            "has_alerts": len(found_alerts) > 0,
            "alerts": found_alerts,
            "risk_level": "High" if len(found_alerts) > 1 else "Medium" if len(found_alerts) == 1 else "Low"
        }

    def _get_summary(self, mol) -> Dict:
        """Generate a summary for display."""
        lipinski = self._check_lipinski(mol)
        drug_likeness = self._assess_drug_likeness(mol)
        bbb = self._predict_bbb_permeability(mol)
        
        return {
            "verdict": drug_likeness["category"],
            "score": drug_likeness["score"],
            "lipinski_violations": lipinski["violations"],
            "oral_bioavailability": "Good" if lipinski["passed"] else "Poor",
            "bbb_permeant": bbb["permeable"]
        }


def calculate_drug_properties(smiles: str) -> Dict:
    """Convenience function to calculate all drug properties."""
    calculator = DrugPropertiesCalculator()
    return calculator.calculate_all(smiles)
