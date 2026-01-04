"""
SMILES to PDBQT Converter Service
Converts SMILES strings to 3D PDBQT format for docking
"""
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from meeko import MoleculePreparation, PDBQTWriterLegacy
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# --- RETRY HELPER ---
import time
from functools import wraps

def convert_with_retry(conversion_func, content: str, filename: str, max_retries: int = 3):
    """
    Retry wrapper for conversion functions.
    Handles transient failures with exponential backoff.
    
    Args:
        conversion_func: Function that takes (content, filename) and returns (result, error)
        content: File content to convert
        filename: Original filename
        max_retries: Maximum retry attempts (default 3)
    
    Returns:
        Tuple[Optional[str], Optional[str]]: (converted_content, error_message)
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result, err = conversion_func(content, filename)
            
            if not err and result:
                if attempt > 0:
                    logger.info(f"✅ Conversion succeeded on attempt {attempt + 1}")
                return result, None
            
            # Conversion returned an error
            last_error = err or "Conversion returned empty result"
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 0.5  # 0.5s, 1s, 1.5s
                logger.warning(f"⚠️ Conversion attempt {attempt + 1} failed: {last_error}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 0.5
                logger.warning(f"⚠️ Conversion exception on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Conversion failed after {max_retries} attempts: {e}")
    
    return None, f"Conversion failed after {max_retries} attempts: {last_error}"


def smiles_to_pdbqt(smiles: str, name: str = "ligand") -> Tuple[Optional[str], Optional[str]]:
    """
    Convert a SMILES string to PDBQT format (Hardened Wrapper).
    """
    return smiles_to_pdbqt_hardened(smiles, name)


# --- NEW: Fix 1 & 2 (Hardened Conversion) ---
def check_ligand_properties(mol, is_pdb_file: bool = False) -> Tuple[bool, str]:
    """
    Validate ligand properties for docking suitability.
    NOTE: Fragment check removed - we now auto-select the largest fragment.
    """
    if not mol: return False, "Null molecule"
    
    # 1. Atom Count (3-150 heavy atoms is typical for Vina)
    num_heavy = mol.GetNumHeavyAtoms()
    if num_heavy < 3: return False, "Molecule too small (<3 heavy atoms)"
    if num_heavy > 150: return False, "Molecule too large (>150 heavy atoms)"
    
    # 2. Net Charge (Vina prefers neutral/near-neutral)
    charge = Chem.GetFormalCharge(mol)
    if not (-5 <= charge <= 5):
        return False, f"Net charge {charge} is too extreme"
        
    return True, ""

def validate_pdbqt_quality(pdbqt_string: str, is_receptor: bool = False) -> Tuple[bool, str]:
    """
    Validate PDBQT output quality before returning.
    Prevents malformed PDBQT from reaching Vina.
    """
    if not pdbqt_string:
        return False, "Empty PDBQT string"
    
    try:
        lines = pdbqt_string.splitlines()
        atom_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
        
        # 1. Minimum atom count
        min_atoms = 100 if is_receptor else 3
        if len(atom_lines) < min_atoms:
            return False, f"Only {len(atom_lines)} atoms (expected >{min_atoms})"
        
        # 2. Check coordinate columns are parseable
        try:
            first_atom = atom_lines[0]
            x = float(first_atom[30:38].strip())
            y = float(first_atom[38:46].strip())
            z = float(first_atom[46:54].strip())
            
            # Sanity check coordinates (should be reasonable protein/ligand coords)
            if abs(x) > 10000 or abs(y) > 10000 or abs(z) > 10000:
                return False, f"Extreme coordinates detected: ({x}, {y}, {z})"
        except (ValueError, IndexError) as e:
            return False, f"Malformed coordinate columns: {e}"
        
        # 3. Check for required PDBQT components
        has_atom_types = any("ATOM" in l or "HETATM" in l for l in lines)
        if not has_atom_types:
            return False, "Missing ATOM/HETATM records"
        
        return True, ""
    
    except Exception as e:
        return False, f"Validation error: {e}"


def score_fragment(mol) -> float:
    """
    Score fragment quality (higher = better).
    Prefers larger, more connected fragments.
    Adapted from Universal Ligand Toolkit.
    """
    if not mol:
        return 0.0
    
    score = 0.0
    score += mol.GetNumAtoms() * 2  # Prioritize larger fragments
    score += mol.GetNumBonds()  # Prefer more connected structures
    
    # Bonus for non-hydrogen heavy atoms
    heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() != 'H')
    score += heavy_atoms
    
    return score


def clean_protein_aggressive(content: str) -> str:
    """
    Aggressively clean protein structure before RDKit parsing.
    Removes waters, ions, alternate locations, and duplicates.
    Adapted from Universal Protein Handler.
    """
    lines = []
    seen_atoms = set()
    
    for line in content.splitlines():
        # Skip water molecules (HOH, WAT, TIP3, H2O, SOL)
        if any(water in line for water in ['HOH', 'WAT', 'TIP3', 'H2O', 'SOL']):
            continue
        
        # Skip common ions (Na, K, Cl, Br, Mg, Ca, Zn, Fe, Cu)
        if any(ion in line for ion in ['NA ', 'K  ', 'CL ', 'BR ', 'MG ', 'CA ', 'ZN ', 'FE ', 'CU ']):
            continue
        
        # Process ATOM/HETATM lines
        if line.startswith(('ATOM', 'HETATM')):
            # Skip alternate locations (keep only A or blank)
            if len(line) > 16 and line[16] not in [' ', 'A', '1']:
                continue
            
            # Deduplicate atoms by residue + atom name
            atom_key = line[5:26] if len(line) > 26 else line
            if atom_key not in seen_atoms:
                seen_atoms.add(atom_key)
                lines.append(line)
    
    logger.info(f"Protein cleaning: {len(seen_atoms)} unique atoms retained")
    return '\n'.join(lines)


def smiles_to_pdbqt_hardened(smiles: str, name: str = "ligand") -> Tuple[Optional[str], Optional[str]]:
    """
    Hardened 4-Stage Fallback Pipeline for Ligands.
    1. RDKit Strict (With Salt Removal)
    2. RDKit Relaxed (Sanitize=False)
    3. OpenBabel (File Conversion)
    4. Native Text Fallback (Not applicable for SMILES, but for files)
    """
    try:
        # STAGE 1: RDKit Strict
        remover = SaltRemover.SaltRemover()
        mol = Chem.MolFromSmiles(smiles) 
        
        if mol:
            # Strip Salts
            mol = remover.StripMol(mol, dontRemoveEverything=True)
            # Strict Sanitize
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except:
                mol = None # Failed strict
        
        # STAGE 2: RDKit Relaxed
        if not mol:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol:
                mol = remover.StripMol(mol, dontRemoveEverything=True)
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except:
                    pass # Keep going even if partial fail
                    
        if not mol:
             return None, f"Invalid SMILES (RDKit rejected): {smiles[:30]}..."

        # Validate Properties (SMILES mode - strict validation)
        valid, msg = check_ligand_properties(mol, is_pdb_file=False)
        if not valid:
             return None, f"Ligand Validation Failed: {msg}"

        # Setup & Convert (Shared)
        mol = Chem.AddHs(mol)
        
        # 3D Generation
        res = AllChem.EmbedMolecule(mol, randomSeed=42)
        if res != 0:
             res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
             if res != 0: return None, "Failed to generate 3D coordinates"
             
        # Optimize
        try: AllChem.MMFFOptimizeMolecule(mol)
        except: pass
        
        # Meeko
        preparator = MoleculePreparation()
        setups = preparator.prepare(mol)
        if setups:
            pdbqt_out = PDBQTWriterLegacy.write_string(setups[0])[0]
            # Validate ligand output
            is_valid, val_msg = validate_pdbqt_quality(pdbqt_out, is_receptor=False)
            if not is_valid:
                logger.warning(f"Ligand PDBQT validation failed: {val_msg}")
                return None, f"Generated PDBQT failed validation: {val_msg}"
            return pdbqt_out, None
            
        return None, "Meeko preparation failed"

    except Exception as e:
        return None, f"Hardened Conversion Failed: {e}"


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
        elif ext == 'pdbqt':
            # Treat PDBQT as PDB for RDKit parsing (often works for atoms)
            logger.info(f"Parsing PDBQT input: {filename}")
            stripped_content = ""
            for line in content.splitlines():
                if line.startswith(('ATOM', 'HETATM')):
                     stripped_content += line[:66] + "\n"
            mol = Chem.MolFromPDBBlock(stripped_content, removeHs=False)
            if not mol:
                mol = Chem.MolFromPDBBlock(stripped_content, removeHs=False, sanitize=False)

        # --- FALLBACK: If RDKit failed, try OpenBabel via file_converter (ONLY IF INSTALLED) ---
        import shutil
        has_obabel = shutil.which('obabel') is not None
        
        if not mol and has_obabel:
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
                 # Don't fail yet, try pass-through for PDBQT
                 pass

        # --- FINAL FALLBACK: Pass-Through for PDBQT ---
        if not mol and ext == 'pdbqt':
             logger.warning(f"All parsers failed for {filename}. Attempting pass-through validation.")
             valid, msg = validate_pdbqt_quality(content)
             if valid:
                  return content, None
             else:
                  return None, f"Could not parse PDBQT and validation failed: {msg}"
        
        if not mol:
             return None, f"Could not parse molecule format: {ext} (RDKit failed, Obabel {'missing' if not has_obabel else 'failed'})"
        
        # FIX 1: Strip salts and validate for Files (same as SMILES)
        if mol:
             try:
                  remover = SaltRemover.SaltRemover()
                  mol = remover.StripMol(mol, dontRemoveEverything=True)
                  logger.info(f"Salt removal applied to {filename}")
                  
                  # If still multiple fragments after salt removal, select largest
                  frags = Chem.GetMolFrags(mol, asMols=True)
                  if len(frags) > 1:
                       logger.warning(f"{filename} has {len(frags)} fragments after salt removal. Selecting best fragment.")
                       # Score and select best fragment
                       scored_frags = [(score_fragment(f), f) for f in frags]
                       best_score, mol = max(scored_frags, key=lambda x: x[0])
                       logger.info(f"Selected fragment with {mol.GetNumAtoms()} atoms (score: {best_score:.1f})")
                       
             except Exception as e:
                  logger.warning(f"Salt removal/fragment selection failed for {filename}: {e}")
             
             valid, msg = check_ligand_properties(mol, is_pdb_file=True)
             if not valid:
                  # Log warning but don't fail - let Vina handle edge cases
                  logger.warning(f"File Validation Warning for {filename}: {msg}")
                  # Only hard-fail on extreme cases (>5 fragments suggests corruption)
                  if "fragments" in msg:
                       frag_count = len(Chem.GetMolFrags(mol))
                       if frag_count > 5:
                            return None, f"File Validation Failed: {msg} (likely corrupted file)"

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
        # CRITICAL: Select largest fragment RIGHT BEFORE Meeko to avoid "10 fragments" error
        frags_before_meeko = Chem.GetMolFrags(mol, asMols=True)
        if len(frags_before_meeko) > 1:
            logger.warning(f"{filename}: Meeko pre-check found {len(frags_before_meeko)} fragments. Selecting largest.")
            scored_frags = [(score_fragment(f), i, f) for i, f in enumerate(frags_before_meeko)]
            best_score, frag_idx, mol = max(scored_frags, key=lambda x: x[0])
            logger.info(f"Pre-Meeko: Selected fragment {frag_idx} ({mol.GetNumAtoms()} atoms, score: {best_score:.1f})")
            # Re-add hydrogens to selected fragment
            mol = Chem.AddHs(mol, addCoords=True)
        
        preparator = MoleculePreparation()
        setups = preparator.prepare(mol)
        if setups:
            result = PDBQTWriterLegacy.write_string(setups[0])
            pdbqt_string = result[0] if isinstance(result, tuple) else result
        else:
            return None, "Meeko preparation failed to generate a molecule setup"
        
        # Final validation before returning
        is_valid, val_msg = validate_pdbqt_quality(pdbqt_string, is_receptor=False)
        if not is_valid:
            return None, f"PDBQT validation failed: {val_msg}"
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
            
            # Heuristic: Detect if input already has multiple models or huge header
            has_protein = False
            
            for line in lines:
                if line.startswith("ATOM  "):
                     # Check alternate location indicator (column 16, index 16 in 0-indexed string)
                     # Keep only 'A' or ' ' alternative locations to avoid duplicates
                     alt_loc = line[16] if len(line) > 16 else ' '
                     if alt_loc not in [' ', 'A', '1']:
                         continue
                     cleaned_lines.append(line)
                     has_protein = True
                elif line.startswith("TER"):
                     cleaned_lines.append(line)
                # EXPLICITLY SKIP HETATM (Waters, Ligands, Ions) for Receptor Prep
                # This guarantees we only get the peptide chain, solving the "10 fragments" Meeko crash.
                elif line.startswith("HETATM"):
                     continue # Aggressive skip
            
            if not has_protein:
                 # Fallback if we accidentally stripped everything (e.g. valid HETATM protein like peptides?)
                 # Just use original content
                 cleaned_content = content
            else:
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
            # Remove Waters (Optional but recommended)
            try:
                water = Chem.MolFromSmarts('[OH2]')
                if water:
                    mol = Chem.DeleteSubstructs(mol, water)
            except: pass

            # FIX: Select largest fragment if multiple (handles ions/salts)
            frags = Chem.GetMolFrags(mol, asMols=True)
            if len(frags) > 1:
                logger.warning(f"{filename}: {len(frags)} fragments detected. Selecting largest.")
                scored_frags = [(score_fragment(f), i, f) for i, f in enumerate(frags)]
                best_score, frag_idx, mol = max(scored_frags, key=lambda x: x[0])
                logger.info(f"Selected fragment {frag_idx} ({mol.GetNumAtoms()} atoms, score: {best_score:.1f})")

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
            # Fall through to Layer 3
            # --- LAYER 2.5 DISABLED ---
            # User Feedback: "when protein breaks in pices, run docking job with simple crystaline form"
            # Layer 2.5 relies on RDKit atoms, which might be "6 fragments".
            # Layer 3 relies on raw text parsing, which treats everything as one block.
            # We disable Layer 2.5 to force the simpler, more robust Layer 3.
            pass

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
                
                # --- Parsing Strategy: Hybrid (Fixed Width -> Split) ---
                x, y, z = 0.0, 0.0, 0.0
                parsing_method = "fixed"
                
                try:
                    # Clean input line
                    clean_line = line.replace('\t', ' ')
                    
                    # 1. Try Standard Fixed Width (Most Reliable for metadata)
                    try:
                        rec_name = clean_line[0:6].strip()
                        serial = int(clean_line[6:11].strip()) if clean_line[6:11].strip() else atom_cnt
                        name = clean_line[12:16].strip()
                        alt_loc = clean_line[16:17] if len(clean_line) > 16 else ' '
                        res_name = clean_line[17:20].strip()
                        chain_id = clean_line[21:22] if len(clean_line) > 21 else 'A'
                        res_seq = int(clean_line[22:26].strip()) if clean_line[22:26].strip() else 1
                        icode = clean_line[26:27] if len(clean_line) > 26 else ' '
                        
                        x = float(clean_line[30:38])
                        y = float(clean_line[38:46])
                        z = float(clean_line[46:54])
                        
                        # Official Element
                        element = ""
                        if len(clean_line) >= 78:
                             element = clean_line[76:78].strip()
                             
                    except (ValueError, IndexError):
                        # 2. Fallback: Split Strategy (Handles tabs/shifts)
                        parsing_method = "split"
                        parts = line.split()
                        # ATOM(0) Serial(1) Name(2) Res(3) Chain(4) ResSeq(5) X(6) Y(7) Z(8) ...
                        # This is risky but often works for messy files
                        # Heuristic: Scan for 3 consecutive floats
                        coords_found = False
                        float_buffer = []
                        res_name_guess = "UNK"
                        name_guess = parts[2] if len(parts) > 2 else "C"
                        
                        for i, p in enumerate(parts):
                            try:
                                val = float(p)
                                # Check if it looks like a coordinate (-1000 to 1000, usually has dot)
                                if '.' in p:
                                    float_buffer.append(val)
                            except:
                                float_buffer = [] # Reset on non-float
                            
                            if len(float_buffer) == 3:
                                x, y, z = float_buffer
                                coords_found = True
                                # Metadata is hard to guess, stick to defaults/guesses
                                name = name_guess
                                res_name = res_name_guess
                                if i >= 8: # If coords are late, earlier stuff might be valid
                                    pass 
                                break
                        
                        if not coords_found:
                            raise ValueError("Could not find coordinates")

                        # Defaults for split mode
                        serial = atom_cnt
                        res_seq = 1
                        chain_id = 'A'
                        alt_loc = ' '
                        icode = ' '
                        element = "" # Will guess later

                    # --- Element Guessing (Shared) ---
                    # 1. From Official (if fixed parsed)
                    # 2. Heuristic from Name
                    if not element and name:
                        import re
                        alpha_only = re.sub(r'[^A-Za-z]', '', name)
                        if alpha_only:
                            if len(alpha_only) >= 2 and alpha_only[1].islower():
                                element = alpha_only[:2] 
                            else:
                                element = alpha_only[0]
                        else:
                            element = "C"
                            
                    element = element.upper()
                    if not element: element = "C"

                    # --- AutoDock Type Mapping ---
                    valid_types = {
                        'H','C','N','O','F','P','S','CL','BR','I','MG','CA','FE','ZN','MN','NA','K'
                    }
                    
                    ad_type = atom_map.get(element, element)
                    if element == 'C': ad_type = 'C' 
                    
                    if ad_type not in valid_types:
                         if element == 'K': ad_type = 'K'
                         else: ad_type = 'C' 
                        
                    atom_cnt += 1
                    
                    # --- Reconstruction ---
                    fmt_name = f"{name:<3}" if len(name)<4 else name[:4]
                    if len(name) <= 3 and not name[0].isdigit(): 
                         fmt_name = f" {name:<3}"
                    else:
                         fmt_name = f"{name:<4}"

                    line_prefix = (
                        f"{'ATOM':<6}{serial:>5} {fmt_name}{alt_loc}{res_name:>3} {chain_id}{res_seq:>4}{icode}   "
                    )
                    
                    x_str = f"{x:8.3f}"
                    y_str = f"{y:8.3f}"
                    z_str = f"{z:8.3f}"
                    
                    newline = f"{line_prefix}{x_str}{y_str}{z_str}  1.00  0.00    0.000 {ad_type:<2}"
                    lines.append(newline)
                except Exception as line_err:
                    continue

        pdbqt_string = "\n".join(lines)
        
        # Validate result isn't empty
        if len(pdbqt_string) > 10 and atom_cnt > 0:
            logger.info("Layer 3 (Native Text Fallback) Success.")
            return pdbqt_string, None
            
    except Exception as e:
        logger.error(f"Layer 3 (Native Text Fallback) failed: {e}")
        # Final pass through...

    # --- LAYER 4: Pass-Through (The "Just Work" Logic) ---
    # User Request: "when protein breaks in pices, run docking job with simple crystaline form submitted by user dont give error"
    # If all sophisticated parsing failed, we assume the user uploaded a valid PDB/PDBQT and just return it (or minimal conversion).
    
    logger.warning(f"All conversion layers failed for {filename}. Engaging Layer 4 Pass-Through.")
    
    if len(content) > 10:
        if ext == 'pdbqt':
            return content, None
        
        # If PDB, try to just strip headers and return as PDBQT (Vina often accepts PDB-like atoms if space separated)
        # Or better, just convert atoms blindly.
        if ext in ['pdb', 'ent']:
             # Last ditch: Native Writer on raw text lines without RDKit
             try:
                 lines = []
                 for line in content.splitlines():
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        # Ensure it fits minimum width
                        if len(line) > 54:
                             lines.append(line)
                 
                 if len(lines) > 0:
                     return "\n".join(lines), None
             except:
                 pass

        return content, None  # Total Hail Mary

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
