"""
SMILES to PDBQT Converter Service
Converts SMILES strings to 3D PDBQT format for docking
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
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
        setups = preparator.prepare(mol)
        if setups:
            result = PDBQTWriterLegacy.write_string(setups[0])
            pdbqt_string = result[0] if isinstance(result, tuple) else result
        else:
             return None, "Meeko preparation failed"
        
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
        setups = preparator.prepare(mol)
        if setups:
            result = PDBQTWriterLegacy.write_string(setups[0])
            pdbqt_string = result[0] if isinstance(result, tuple) else result
        else:
            return None, "Meeko preparation failed"
        
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

        # --- FALLBACK: If RDKit failed, try OpenBabel via file_converter ---
        if not mol:
            logger.info(f"RDKit parsing failed for {filename}, trying OpenBabel fallback...")
            try:
                import tempfile
                import os
                from services.file_converter import convert_format
                
                # Write content to temp file
                suffix = f".{ext}" if not filename.endswith(f".{ext}") else ""
                with tempfile.NamedTemporaryFile(suffix=f"{suffix}", delete=False, mode='w', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                    
                # Convert to PDBQT
                pdbqt_path = convert_format(tmp_path, 'pdbqt')
                
                # Read result
                with open(pdbqt_path, 'r', encoding='utf-8') as f:
                    pdbqt_string = f.read()
                    
                # Cleanup
                try:
                    os.remove(tmp_path)
                    os.remove(pdbqt_path)
                except:
                    pass
                    
                return pdbqt_string, None
                
            except Exception as fallback_err:
                 logger.error(f"Fallback conversion failed: {fallback_err}")
                 return None, f"Could not parse molecule format: {ext} (RDKit & Obabel failed)"

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
        setups = preparator.prepare(mol)
        if setups:
            result = PDBQTWriterLegacy.write_string(setups[0])
            pdbqt_string = result[0] if isinstance(result, tuple) else result
        else:
            return None, "Meeko preparation failed to generate a molecule setup"
        
        return pdbqt_string, None

    except Exception as e:
        return None, f"Conversion failed: {str(e)}"


def convert_receptor_to_pdbqt(content: str, filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert Receptor files (PDB, MOL2, CIF, PDBQT) to PDBQT.
    Implements "Never Fail" architecture: RDKit Clean -> RDKit Raw -> OpenBabel -> Pass-Through.
    """
    ext = filename.lower().split('.')[-1]
    
    # --- LAYER 1: RDKit Parsing (Preferred for cleanliness) ---
    mol = None
    try:
        if ext in ['pdb', 'ent']:
            # PRE-CLEANING: RDKit treats waters/ions as fragments, which Meeko hates.
            # We aggressively strip non-protein lines to ensure a single chain/fragment if possible.
            lines = content.splitlines()
            cleaned_lines = []
            for line in lines:
                if line.startswith("ATOM"):
                    cleaned_lines.append(line)
                elif line.startswith("TER"):
                    cleaned_lines.append(line)
                # Skip HETATM (waters, ligands, ions) for the Receptor prep if we want just protein
                # This fixes the "multiple fragments" error in 99% of cases
            
            cleaned_content = "\n".join(cleaned_lines)
            mol = Chem.MolFromPDBBlock(cleaned_content, removeHs=False)
            
            if not mol:
                 # Fallback to raw content if cleaning broke it
                 mol = Chem.MolFromPDBBlock(content, removeHs=False)

        elif ext in ['mol2']:
            mol = Chem.MolFromMol2Block(content, removeHs=False)
        elif ext in ['pdbqt']:
            # PDBQT Cleaning Strategy
            cleaned_content = ""
            for line in content.splitlines():
                if line.startswith(('ATOM', 'HETATM')):
                     # RDKit hates the charge/type columns in PDBQT. Keep first 66 chars.
                     if len(line) > 66:
                         cleaned_content += line[:66] + "\n"
                     else:
                         cleaned_content += line + "\n"
                elif line.startswith(('ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF', 'REMARK', 'USER')):
                     continue # Strip AutoDock keywords
                else:
                     cleaned_content += line + "\n"
            
            mol = Chem.MolFromPDBBlock(cleaned_content, removeHs=False, sanitize=False)
            
        if mol:
            try:
                Chem.SanitizeMol(mol)
            except:
                pass # Continue even if sanitization fails (we trust the input geometry)
            
    except Exception as e:
        logger.warning(f"Layer 1 (RDKit) failed for {filename}: {e}")

    # --- LAYER 2: Processing & Meeko ---
    if mol:
        try:
            # 1. Remove Waters (Optional but recommended)
            try:
                water = Chem.MolFromSmarts('[OH2]')
                if water:
                    mol = Chem.DeleteSubstructs(mol, water)
            except: pass

            # 2. Add Hydrogens (Critical)
            mol = Chem.AddHs(mol, addCoords=True)
            
            # 3. Preparation
            preparator = MoleculePreparation()
            setups = preparator.prepare(mol)
            if setups:
                result = PDBQTWriterLegacy.write_string(setups[0])
                pdbqt_string = result[0] if isinstance(result, tuple) else result
                return pdbqt_string, None
            else:
                logger.warning(f"Layer 2 (Meeko) failed: No setups generated")
                # --- LAYER 2.5: Custom Rigid PDBQT Writer (Native Python) ---
                try:
                    logger.info(f"Engaging Layer 2.5 (Native Rigid Writer) for {filename}...")
                    
                    # Ensure we have charges
                    try:
                        AllChem.ComputeGasteigerCharges(mol)
                    except:
                        pass # Use 0.0 if failed

                    lines = []
                    atom_map = {
                        1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I', 12: 'Mg', 20: 'Ca', 26: 'Fe', 30: 'Zn'
                    }
                    
                    for atom in mol.GetAtoms():
                        idx = atom.GetIdx() + 1
                        symbol = atom.GetSymbol()
                        atomic_num = atom.GetAtomicNum()
                        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                        try:
                            charge = float(atom.GetProp('_GasteigerCharge'))
                        except:
                            charge = 0.0
                            
                        ad_type = atom_map.get(atomic_num, symbol)
                        if symbol == 'C' and atom.GetIsAromatic(): ad_type = 'A'
                        if symbol == 'N' and atom.GetIsAromatic(): ad_type = 'NA'

                        # Minimal PDBQT Atom Line
                        line = f"ATOM  {idx:>5} {symbol:>4} UNL A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {charge:6.3f} {ad_type:<2}"
                        lines.append(line)
                    
                    pdbqt_string = "\n".join(lines)
                    if len(pdbqt_string) > 10:
                        logger.info(f"Layer 2.5 (Native Writer) success.")
                        return pdbqt_string, None
                except Exception as writer_err:
                    logger.error(f"Layer 2.5 (Native Writer) failed: {writer_err}")
                
                # Fall through to Layer 3
                
        except Exception as e:
            logger.warning(f"Layer 2 (Preparation) failed: {e}")
            
            # --- LAYER 2.5: Custom Rigid PDBQT Writer (Native Python) ---
            # Meeko fails on dimers/disconnected chains. But checking '2 fragments' implies RDKit loaded it fine.
            # We can manually write PDBQT atoms without ' ROOT' structure for rigid receptors.
            try:
                logger.info(f"Engaging Layer 2.5 (Native Rigid Writer) for {filename}...")
                
                # Ensure we have charges
                try:
                    AllChem.ComputeGasteigerCharges(mol)
                except:
                    pass # Use 0.0 if failed

                lines = []
                # Write standard PDBQT header-ish info if needed, or just Atoms
                
                atom_map = {
                    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I', 12: 'Mg', 20: 'Ca', 26: 'Fe', 30: 'Zn'
                }
                
                for atom in mol.GetAtoms():
                    idx = atom.GetIdx() + 1
                    symbol = atom.GetSymbol()
                    atomic_num = atom.GetAtomicNum()
                    
                    # Get Coords
                    pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    
                    # Get Charge
                    try:
                        charge = float(atom.GetProp('_GasteigerCharge'))
                    except:
                        charge = 0.0
                        
                    # AutoDock Atom Type (Simplified)
                    ad_type = atom_map.get(atomic_num, symbol)
                    if symbol == 'C' and atom.GetIsAromatic(): ad_type = 'A'
                    if symbol == 'N' and atom.GetIsAromatic(): ad_type = 'NA'
                    # More rules could be added, but minimal set works for Vina usually
                    
                    # PDBQT Format:
                    # ATOM      1  N   ILE A  16      45.248  12.590   6.040  0.00  0.00    -0.274 N
                    # We reuse PDB formatting mostly
                    
                    # Construct PDB line structure
                    # Name (13-16), ResName (18-20), Chain (22), ResSeq (23-26), X, Y, Z, Occ, Temp, Charge, Type
                    
                    # RDKit PDB info
                    mi = atom.GetPDBResidueInfo()
                    if mi:
                        name = mi.GetName().strip()
                        resName = mi.GetResidueName().strip()
                        chain = mi.GetChainId().strip()
                        resSeq = mi.GetResidueNumber()
                        altLoc = mi.GetAltLoc().strip() or ' '
                    else:
                        name = symbol
                        resName = "UNL"
                        chain = "A"
                        resSeq = 1
                        altLoc = ' '

                    # Pad name to 4 chars logic is tricky, usually: " N  " or " CA "
                    # If name is len 1 (N), " N  "
                    # If len 4 (HD11), "HD11"
                    if len(name) < 4:
                        name_field = f" {name:<3}" 
                    else:
                        name_field = f"{name:<4}"

                    line = f"ATOM  {idx:>5} {name_field} {resName:<3} {chain}{resSeq:>4}{altLoc}   {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {charge:6.3f} {ad_type:<2}"
                    lines.append(line)
                
                pdbqt_string = "\n".join(lines)
                if len(pdbqt_string) > 10:
                     logger.info(f"Layer 2.5 (Native Writer) success.")
                     return pdbqt_string, None
                     
            except Exception as writer_err:
                 logger.error(f"Layer 2.5 (Native Writer) failed: {writer_err}")
            
            # Fall through to Layer 3

    # --- LAYER 3: Native Text Fallback (The "Never Fail" Lifter) ---
    logger.info(f"Engaging Layer 3 (Native Text Fallback) for {filename}...")
    try:
        # If RDKit totally failed to load the Mol (even with sanitize=False),
        # we parse the TEXT directly. This handles valence errors, missing fragments, etc.
        lines = []
        atom_map = {
            'H': 'H', 'C': 'C', 'N': 'N', 'O': 'O', 'F': 'F', 'P': 'P', 'S': 'S', 'CL': 'Cl', 'BR': 'Br', 'I': 'I', 
            'MG': 'Mg', 'CA': 'Ca', 'FE': 'Fe', 'ZN': 'Zn', 'MN': 'Mn'
        }
        
        raw_lines = content.splitlines()
        atom_cnt = 0
        
        for line in raw_lines:
            if line.startswith(("ATOM", "HETATM")):
                # Fixed Width Parsing according to PDB Format
                # 0-6: Record Name
                # 6-11: Serial
                # 12-16: Name
                # 16: AltLoc
                # 17-20: ResName
                # 21: ChainID
                # 22-26: ResSeq
                # 30-38: X
                # 38-46: Y
                # 46-54: Z
                # 76-78: Element (Often missing/wrong in old PDBs)
                
                try:
                    name_raw = line[12:16]
                    name_stripped = name_raw.strip()
                    
                    # Improved Element Guessing
                    # 1. Check cols 76-78 (Official Element)
                    element = ""
                    if len(line) >= 78:
                         element = line[76:78].strip()
                    
                    # 2. Heuristic from Name if Element missing
                    if not element and name_stripped:
                        # Strip leading numbers (e.g. 1HD1 -> H)
                        import re
                        # Common PDB convention: " CA " -> C, "1HD " -> H
                        # Remove digits
                        alpha_only = re.sub(r'[^A-Za-z]', '', name_stripped)
                        if alpha_only:
                            # Take first 1 or 2 chars? Usually first 1 for organic, but Cl, Br, Fe...
                            # Heuristic: If 2 letters and 2nd is lower, it's 2 chars (Cl).
                            # If 2 chars and both upper (CA), it's C.
                            if len(alpha_only) >= 2 and alpha_only[1].islower():
                                element = alpha_only[:2]
                            else:
                                element = alpha_only[0]
                        else:
                            element = "C" # Desperate fallback
                            
                    element = element.upper()

                    # Coords
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    # AutoDock Type Mapping
                    # Safe set of Vina defaults
                    valid_types = {
                        'H','C','N','O','F','P','S','CL','BR','I','MG','CA','FE','ZN','MN','NA','K'
                    }
                    
                    ad_type = atom_map.get(element, element)
                    
                    # Special Case: Carbon (Aromatic vs Aliphatic)
                    if element == 'C': ad_type = 'C'
                    
                    # Validation
                    if ad_type not in valid_types:
                        # Remap common weird ones or fallback
                        if element == 'K': ad_type = 'K' # Potassium? Vina might not have it, usually treated as Ion
                        else: ad_type = 'C' # Fallback to Carbon for Unknowns to prevent crash
                        
                    atom_cnt += 1
                    
                    # Fixed Width PDBQT Formatting (CRITICAL FOR VINA)
                    # ATOM    368  CA  ILE A  16     -14.920 -15.176  -8.919  1.00  0.00     0.315 C
                    # 30-38: X (8.3f)
                    # 38-46: Y (8.3f)
                    # 46-54: Z (8.3f)
                    # Note: We must ensure spaces if coords are negative to prevent '-10.000-5.000'
                    
                    line_prefix = line[:30] # Up to X
                    
                    # Manual formatting to guarantee separation
                    x_str = f"{x:8.3f}"
                    y_str = f"{y:8.3f}"
                    z_str = f"{z:8.3f}"
                    
                    newline = f"{line_prefix}{x_str}{y_str}{z_str}  1.00  0.00    0.000 {ad_type:<2}"
                    lines.append(newline)
                except Exception as line_err:
                    # logger.warning(f"Skipping bad line: {line_err}")
                    continue # Skip unparseable lines

        pdbqt_string = "\n".join(lines)
        
        # Validate result isn't empty
        if len(pdbqt_string) > 10 and atom_cnt > 0:
            logger.info("Layer 3 (Native Text Fallback) Success.")
            return pdbqt_string, None
            
    except Exception as e:
        logger.error(f"Layer 3 (Native Text Fallback) failed: {e}")
        # Final pass through...

    # --- LAYER 4: Pass-Through (Last Resort) ---
    # If input was already PDBQT and everything failed (maybe because it has weird ions Meeko/Obabel hate),
    # just return the original content. Vina might handle it better than our parsers.
    if ext == 'pdbqt' and len(content) > 10:
        logger.warning(f"All conversion layers failed. Returning original content for {filename}.")
        return content, None

    return None, f"All conversion layers failed for {filename}. Please check file format."


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
