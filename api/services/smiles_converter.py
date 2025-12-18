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

def pdb_to_pdbqt(pdb_content: str, remove_water: bool = True, add_hydrogens: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert PDB content to PDBQT string using Meeko.
    Suitable for Receptor and Ligand preparation.
    """
    try:
        # 1. Read PDB Block
        mol = Chem.MolFromPDBBlock(pdb_content, removeHs=False)
        if not mol:
            # Try cleaning up basic issues if strict parsing fails
            mol = Chem.MolFromPDBBlock(pdb_content, removeHs=False, sanitize=False)
            if mol:
                 try:
                     Chem.SanitizeMol(mol)
                 except:
                     pass # Best effort

        if not mol:
            return None, "Failed to parse PDB content"

        # 2. Preparation steps
        if add_hydrogens:
            mol = Chem.AddHs(mol, addCoords=True)
        
        # Note: RDKit RemoveHs removes ALL Hs, but we want to remove implicit only or handle specific cases? 
        # Actually RDKit by default reads usually without Hs unless specified. 
        # AddHs(addCoords=True) is good.

        # 3. Convert with Meeko
        preparator = MoleculePreparation()
        
        # If it's a receptor, we might want specific settings, but Meeko handles generic molecules well.
        # For receptor docking, usually we need to preserve specific structure. 
        # Meeko is primarily for Ligands.
        # However, for simple PDBQT conversion of protein with explicit Hs, it works reasonable well for Vina/Gnina.
        
        preparator.prepare(mol)
        pdbqt_string = preparator.write_pdbqt_string()
        
        return pdbqt_string, None
        
    except Exception as e:
        return None, f"PDB Conversion error: {str(e)}"


def convert_to_pdbqt(content: str, filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generic converter that handles SMILES (if .smi), PDB, or SDF content.
    Returns (pdbqt_content, error_message).
    """
    ext = filename.lower().split('.')[-1]
    
    try:
        mol = None
        
        # 1. Parse based on extension
        if ext in ['pdb', 'ent']:
            mol = Chem.MolFromPDBBlock(content, removeHs=False)
        elif ext in ['sdf', 'mol', 'sd']:
            # RDKit MolFromMolBlock for SDF
            mol = Chem.MolFromMolBlock(content, removeHs=False)
            if not mol:
                # Try sanitization off
                mol = Chem.MolFromMolBlock(content, removeHs=False, sanitize=False)
        elif ext in ['mol2']:
             mol = Chem.MolFromMol2Block(content, removeHs=False)
             if not mol:
                 mol = Chem.MolFromMol2Block(content, removeHs=False, sanitize=False)

        if not mol:
             return None, f"Could not parse molecule format: {ext}"

        # 2. Add Hydrogens (3D requires them, usually)
        mol = Chem.AddHs(mol, addCoords=True)
        
        # 3. Generate 3D Coords if missing (SDF usually has them, but safety check)
        # Check if we have conformers
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)

        # 4. Meeko Preparation
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        pdbqt_string = preparator.write_pdbqt_string()
        
        return pdbqt_string, None

    except Exception as e:
        return None, f"Conversion failed: {str(e)}"


def convert_receptor_to_pdbqt(content: str, filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert Receptor files (PDB, MOL2, CIF, PDBQT) to PDBQT.
    Receptors MUST have 3D coordinates. We do not generate them.
    """
    ext = filename.lower().split('.')[-1]
    
    try:
        mol = None
        
        if ext in ['pdb', 'ent']:
            mol = Chem.MolFromPDBBlock(content, removeHs=False)
        elif ext in ['mol2']:
            mol = Chem.MolFromMol2Block(content, removeHs=False)
        elif ext in ['cif', 'mmcif']:
            # Try parsing MMCIF
            mol = Chem.MolFromMMCIFBlock(content)
        elif ext in ['pdbqt']:
            # Trust input if it's already PDBQT
            return content, None
            
        if not mol:
             return None, f"Could not parse receptor format: {ext}"

        # Check for 3D coordinates
        if mol.GetNumConformers() == 0:
            return None, "Receptor file missing 3D coordinates"

        # 1. Remove Waters (HOH) - Common cause of fragmentation
        try:
             # pattern for water
             water = Chem.MolFromSmarts('[OH2]')
             if water:
                 mol = Chem.DeleteSubstructs(mol, water)
                 Chem.SanitizeMol(mol)
        except Exception:
             pass # Continue if water removal fails
        
        # 2. Add Hydrogens (Critical for binding pockets)
        # Note: AddHs might add H to salts/ions making them weird, but needed for protein.
        mol = Chem.AddHs(mol, addCoords=True)
        
        # 3. Handle Fragmentation (Chains, Ions, etc.)
        # Meeko fails if multiple disconnected fragments exist.
        # Strategy: Get fragments, keep large ones (protein chains), process individually, then merge PDBQT.
        
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        total_pdbqt_lines = []
        
        print(f"DEBUG: Receptor decomposed into {len(frags)} fragments.")
        
        for frag in frags:
            # Filter out tiny fragments (e.g. single ions like Na+, Cl- or remaining waters)
            if frag.GetNumAtoms() < 4: 
                continue
                
            try:
                # Prepare fragment
                preparator = MoleculePreparation()
                preparator.prepare(frag)
                frag_pdbqt = preparator.write_pdbqt_string()
                
                # Flatten (Rigidify)
                for line in frag_pdbqt.splitlines():
                    if line.startswith(('ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF', 'REMARK')):
                        continue
                    total_pdbqt_lines.append(line)
                    
            except Exception as e:
                print(f"Warning: Failed to convert a fragment: {e}")
                continue
        
        if not total_pdbqt_lines:
             return None, "Receptor conversion produced no valid PDBQT lines (maybe all fragments were too small?)"
             
        rigid_pdbqt = "\n".join(total_pdbqt_lines)
        return rigid_pdbqt, None
        
    except Exception as e:
        return None, f"Receptor conversion failed: {str(e)}"
