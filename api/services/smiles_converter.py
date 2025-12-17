"""
SMILES to PDBQT Converter Service
Converts SMILES strings to 3D PDBQT format for docking
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def smiles_to_pdbqt(smiles: str, name: str = "ligand") -> Tuple[Optional[str], Optional[str]]:
    """
    Convert a SMILES string to PDBQT format.
    
    Args:
        smiles: SMILES string of the molecule
        name: Name for the molecule (used in filename)
        
    Returns:
        Tuple of (pdbqt_string, error_message)
        If successful: (pdbqt_content, None)
        If failed: (None, error_message)
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None, f"Invalid SMILES: {smiles[:50]}..."
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result != 0:
            # Try with different parameters if default fails
            result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
            if result != 0:
                return None, f"Failed to generate 3D coordinates for: {smiles[:50]}..."
        
        # Optimize geometry using MMFF
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            # If MMFF fails, try UFF
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                logger.warning(f"Geometry optimization failed for {name}, using embedded coords")
        
        # Convert to PDBQT using Meeko
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        pdbqt_string = preparator.write_pdbqt_string()
        
        if not pdbqt_string or "ATOM" not in pdbqt_string:
            return None, f"PDBQT generation failed for: {smiles[:50]}..."
        
        return pdbqt_string, None
        
    except Exception as e:
        return None, f"Conversion error: {str(e)}"


def validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False
