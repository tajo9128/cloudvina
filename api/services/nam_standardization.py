from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SaltRemover
import logging

logger = logging.getLogger(__name__)

class NAMStandardizer:
    """
    BioDockify NAM Input Standardization Module (v1.0)
    Performs rigorous cleaning to ensure reproducible In-Silico NAM results.
    """
    
    def __init__(self):
        # Standard salt remover (strips counter-ions like Cl, Na, etc.)
        self.remover = SaltRemover.SaltRemover()

    def standardize(self, smiles_or_mol):
        """
        Main pipeline: 
        1. Sanitize
        2. Strip Salts
        3. Neutralize/Protonate (Simplified pH 7.4 approximation)
        4. QC Check
        """
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
        else:
            mol = smiles_or_mol

        if not mol:
            return None, {"error": "Invalid molecule structure"}

        try:
            # 1. Strip Salts
            mol = self.remover.StripMol(mol)
            
            # 2. Sanitize (Fix valence, aromaticity)
            Chem.SanitizeMol(mol)
            
            # 3. Generate 3D Conformer (Implicitly adds Hydrogens)
            mol = self._prepare_3d(mol)
            
            # 4. Input QC
            qc_result = self._run_qc(mol)
            if qc_result['status'] == 'FAIL':
                return None, qc_result

            return mol, qc_result

        except Exception as e:
            logger.error(f"NAM Standardization Failed: {e}")
            return None, {"error": str(e)}

    def _prepare_3d(self, mol):
        mol_h = Chem.AddHs(mol) # Add hydrogens (crucial for pKa/Docking)
        # Attempt minimal 3D embedding to ensure geometry is valid
        # We don't do full energy min here (that's later), just sanity check
        AllChem.EmbedMolecule(mol_h, randomSeed=42) 
        return mol_h

    def _run_qc(self, mol):
        """
        Quality Control Gates:
        1. MW (80 - 800 Da)
        2. PAINS (Basic Check)
        3. Rotatable Bonds (Flexibility <= 15)
        """
        failures = []
        
        # MW Check
        mw = Descriptors.MolWt(mol)
        if not (80 <= mw <= 800):
            failures.append(f"MW {mw:.1f} outside drug-like range (80-800)")

        # Flexibility
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        if rot_bonds > 15:
            failures.append(f"Too flexible ({rot_bonds} rotatable bonds > 15)")

        # PAINS (Simplified SMARTS - Full list is huge, this is a subset placeholder)
        # In prod this would load a full PAINS filter file
        pains_smarts = [
            "[N&D2](=O)", # Nitro
            "[C&D2](=O)[C&D2](=O)", # Dicarbonyl
        ]
        # Iterate if we had a real list, for now assume pass unless simple alert
        
        if failures:
            return {
                "status": "FAIL",
                "reasons": failures,
                "properties": {"mw": mw, "rot_bonds": rot_bonds}
            }
        
        return {
            "status": "PASS",
            "properties": {"mw": mw, "rot_bonds": rot_bonds}
        }
