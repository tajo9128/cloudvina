"""
Drug Properties Service
Calculates drug-likeness, ADMET predictions, and molecular properties using RDKit.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class DrugPropertiesCalculator:
    """
    Calculates drug-like properties and filters for small molecules.
    Uses RDKit for all calculations - 100% free and open source.
    """
    
    def __init__(self):
        # Lazy import RDKit
        self.pains_catalog = None
        try:
            from rdkit.Chem import FilterCatalog
            from rdkit.Chem.FilterCatalog import FilterCatalogParams
            
            # Initialize PAINS filter
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_catalog = FilterCatalog.FilterCatalog(params)
        except Exception as e:
            logger.warning(f"Failed to initialize PAINS filters (Optional): {e}")
    
    def calculate_all(self, smiles: str) -> Dict:
        """
        Calculate all drug properties for a given SMILES string.
        """
        from rdkit import Chem
        
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
                # Advanced Toxicity (Phase 7 CADD Enhancements)
                "herg": self._predict_herg_liability(mol),
                "ames": self._predict_ames_mutagenicity(mol),
                "cyp": self._predict_cyp_inhibition(mol),
                "admet_links": self._get_admet_links(smiles),
                "summary": self._get_summary(mol)
            }
        except Exception as e:
            logger.error(f"Error calculating properties: {e}")
            return {"error": str(e)}
    
    def _get_molecular_properties(self, mol) -> Dict:
        """Calculate basic molecular properties."""
        from rdkit.Chem import Descriptors, Lipinski
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
        from rdkit.Chem import Descriptors, Lipinski
        """
        Check Lipinski's Rule of 5 for oral bioavailability.
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
        from rdkit.Chem import Descriptors
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
        from rdkit.Chem import Descriptors
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
        if not self.pains_catalog:
             return {"passed": True, "num_alerts": 0, "alerts": [], "note": "PAINS Check Skipped (Filters not loaded)"}
             
        matches = self.pains_catalog.GetMatches(mol)
        pains_alerts = [match.GetDescription() for match in matches]
        
        return {
            "passed": len(pains_alerts) == 0,
            "num_alerts": len(pains_alerts),
            "alerts": pains_alerts[:5]  # Limit to first 5
        }
    
    def _assess_drug_likeness(self, mol) -> Dict:
        lipinski = self._check_lipinski(mol)
        veber = self._check_veber(mol)
        pains = self._check_pains(mol)
        
        # Calculate a simple drug-likeness score (0-100)
        score = 100
        score -= lipinski["violations"] * 15
        score -= veber["violations"] * 10
        score -= pains["num_alerts"] * 20
        score = max(0, min(100, score))
        
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
        from rdkit.Chem import Descriptors, Lipinski
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        hbd = Lipinski.NumHDonors(mol)
        
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
        from rdkit import Chem
        alerts = {
            "Nitro Group": "[N+](=O)[O-]",
            "Hydrazine": "[NX3][NX3]",
            "Michael Acceptor": "[CX3]=[CX3]-[CX3](=[OX1])",
            "Alkyl Halide": "[CX4][F,Cl,Br,I]",
            "Aldehyde": "[CX3H1](=O)[#6]",
            "Epoxide": "C1OC1",
            "Acyl Halide": "[CX3](=[OX1])[F,Cl,Br,I]",
            "Thiourea": "[NX3][CX3](=[SX1])[NX3]",
            "Quinone": "O=C1C=CC(=O)C=C1",
            "Azo Group": "[NX2]=[NX2]"
        }
        
        found_alerts = []
        for name, smarts in alerts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                found_alerts.append(name)
        
        return {
            "has_alerts": len(found_alerts) > 0,
            "alerts": found_alerts,
            "risk_level": "High" if len(found_alerts) > 2 else "Medium" if len(found_alerts) >= 1 else "Low"
        }
    
    def _predict_herg_liability(self, mol) -> Dict:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        
        basic_n_pattern = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]")
        has_basic_n = mol.HasSubstructMatch(basic_n_pattern) if basic_n_pattern else False
        
        risk_factors = []
        if logp > 3.5:
            risk_factors.append("High lipophilicity")
        if has_basic_n:
            risk_factors.append("Basic nitrogen present")
        if mw > 500 and tpsa < 75:
            risk_factors.append("Large hydrophobic molecule")
            
        if len(risk_factors) >= 2:
            risk_level = "High"
            recommendation = "⚠️ Consider hERG patch-clamp assay before proceeding"
        elif len(risk_factors) == 1:
            risk_level = "Moderate"
            recommendation = "Monitor cardiac safety in preclinical studies"
        else:
            risk_level = "Low"
            recommendation = "Favorable hERG profile"
            
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendation": recommendation
        }
    
    def _predict_ames_mutagenicity(self, mol) -> Dict:
        from rdkit import Chem
        mutagenic_alerts = {
            "Aromatic Nitro": "c[N+](=O)[O-]",
            "Aromatic Amine": "c[NH2]",
            "Aromatic Nitroso": "c[N]=O",
            "Polycyclic Aromatic": "c1ccc2c(c1)ccc1ccccc12",
            "Azide": "[N-]=[N+]=[N-]",
            "Diazo": "[N]=[N]"
        }
        
        found_alerts = []
        for name, smarts in mutagenic_alerts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                found_alerts.append(name)
        
        if len(found_alerts) >= 2:
            prediction = "Positive"
            confidence = "High"
        elif len(found_alerts) == 1:
            prediction = "Equivocal"
            confidence = "Moderate"
        else:
            prediction = "Negative"
            confidence = "High"
            
        return {
            "prediction": prediction,
            "confidence": confidence,
            "alerts": found_alerts,
            "recommendation": "AMES test recommended" if prediction != "Negative" else "Low mutagenic risk"
        }
    
    def _predict_cyp_inhibition(self, mol) -> Dict:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        n_aromatic = Descriptors.NumAromaticRings(mol)
        
        cyp_risk = {}
        
        # CYP3A4
        cyp3a4_risk = 0
        if mw > 400: cyp3a4_risk += 1
        if logp > 3: cyp3a4_risk += 1
        if n_aromatic >= 2: cyp3a4_risk += 1
            
        cyp_risk['CYP3A4'] = {
            'inhibition_risk': 'High' if cyp3a4_risk >= 2 else 'Moderate' if cyp3a4_risk == 1 else 'Low',
            'score': cyp3a4_risk
        }
        
        # CYP2D6
        basic_n = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]")
        has_basic_n = mol.HasSubstructMatch(basic_n) if basic_n else False
        
        cyp2d6_risk = 0
        if has_basic_n: cyp2d6_risk += 1
        if n_aromatic >= 1: cyp2d6_risk += 1
            
        cyp_risk['CYP2D6'] = {
            'inhibition_risk': 'High' if cyp2d6_risk >= 2 else 'Moderate' if cyp2d6_risk == 1 else 'Low',
            'score': cyp2d6_risk
        }
        
        # CYP2C9
        cyp2c9_risk = 1 if logp > 2.5 and mw > 300 else 0
        cyp_risk['CYP2C9'] = {
            'inhibition_risk': 'Moderate' if cyp2c9_risk else 'Low',
            'score': cyp2c9_risk
        }
        
        # Overall DDI risk
        total_risk = sum(r['score'] for r in cyp_risk.values())
        ddi_risk = 'High' if total_risk >= 4 else 'Moderate' if total_risk >= 2 else 'Low'
        
        return {
            'isoforms': cyp_risk,
            'overall_ddi_risk': ddi_risk,
            'recommendation': 'CYP inhibition assays recommended' if ddi_risk != 'Low' else 'Low DDI risk'
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
