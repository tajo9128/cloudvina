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
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            # If MMFF fails, try UFF
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
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
        
        # 3. Convert with Meeko
        preparator = MoleculePreparation()
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
            # Use ETKDGv3 for state-of-the-art embedding
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            AllChem.EmbedMolecule(mol, params)

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
    Includes Rigorous Preparation: Water Removal, Ion Preservation, Fragmentation handling.
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
            # Force processing: Try to parse PDBQT as PDB to clean waters/ions
            # RDKit can often read PDBQT atom blocks if treated as PDB
            mol = Chem.MolFromPDBBlock(content, removeHs=False)
            if not mol:
                 # Fallback: If strict parsing fails, trust raw content but warn
                 print(f"Warning: Could not parse PDBQT structure for {filename}. Skipping water removal.")
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
            # Filter: We want to keep Proteins AND Ions, but maybe not isolated weird small things that break Meeko.
            # But "Correct" preparation should keep cofactors.
            # We trust Meeko to handle small fragments if passed individually.
            
            try:
                # Prepare fragment
                preparator = MoleculePreparation()
                preparator.prepare(frag)
                frag_pdbqt = preparator.write_pdbqt_string()
                
                # Flatten (Rigidify) - Strip ROOT/BRANCH/TORSDOF
                # This effectively treats every fragment as a rigid body in the same frame.
                for line in frag_pdbqt.splitlines():
                    if line.startswith(('ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF', 'REMARK')):
                         continue
                    total_pdbqt_lines.append(line)
                    
            except Exception as e:
                # Fallback for single atoms (Ions) that Meeko might choke on
                if frag.GetNumAtoms() == 1:
                     # It's an ion. We can try to manually format it if Meeko failed?
                     # Ideally we just log warning. Most ions pass Meeko fine.
                     print(f"Warning: Failed to convert ion/fragment: {e}")
                else:
                     print(f"Warning: Failed to convert a fragment: {e}")
                continue
        
        if not total_pdbqt_lines:
             return None, "Receptor conversion produced no valid PDBQT lines"
             
        rigid_pdbqt = "\n".join(total_pdbqt_lines)
        return rigid_pdbqt, None
        
    except Exception as e:
        return None, f"Receptor conversion failed: {str(e)}"


def pdbqt_to_pdb(pdbqt_content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert PDBQT content to PDB format using RDKit.
    Useful for MD preparation where OpenMM expects PDB.
    """
    try:
        # PDBQT is close to PDB, but has extra keywords and columns.
        # RDKit often parses it fine if we ignore sanitization initially or treat as PDB.
        # But ROOT/BRANCH lines need stripping for RDKit to be happy? 
        # Actually RDKit ignores unknown lines usually.
        
        mol = Chem.MolFromPDBBlock(pdbqt_content, removeHs=False, sanitize=False)
        if not mol:
            # Fallback: simple text cleanup
            # Strip ROOT, ENDROOT, BRANCH, ENDBRANCH, TORSDOF
            lines = [l for l in pdbqt_content.splitlines() if not l.startswith(('ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF'))]
            cleaned_block = "\n".join(lines)
            mol = Chem.MolFromPDBBlock(cleaned_block, removeHs=False, sanitize=False)
            
        if not mol:
             return None, "Could not parse PDBQT content"

        try:
             Chem.SanitizeMol(mol)
        except:
             pass 
             
        # Write PDB
        pdb_block = Chem.MolToPDBBlock(mol)
        return pdb_block, None
        
    except Exception as e:
        return None, f"PDBQT->PDB conversion failed: {str(e)}"
