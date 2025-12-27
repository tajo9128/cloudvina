"""
ODDT Analysis Service (Deep Science Layer)
Extracts Interaction Fingerprints (IFP) from Protein-Ligand complexes.
Metrics: H-Bonds, Hydrophobic Contacts, Salt Bridges, Pi-Stacking.
"""
import logging
import os
import json
import tempfile
try:
    import oddt
    from oddt import toolkit
    from oddt.interactions import (hbonds, hydrophobic_contacts, salt_bridges, pi_stacking)
except ImportError:
    oddt = None

logger = logging.getLogger("oddt_analysis")

class ODDTAnalysisService:
    """
    Service to perform deep chemical analysis on docking poses.
    """
    
    @staticmethod
    def analyze_interactions(receptor_path: str, ligand_path: str) -> dict:
        """
        Analyze non-covalent interactions between receptor and ligand.
        
        Args:
            receptor_path: Path to receptor PDB/PDBQT
            ligand_path: Path to ligand PDBQT
            
        Returns:
            Dictionary of interaction features.
        """
        if not oddt:
            logger.warning("ODDT not installed. Skipping deep analysis.")
            return {"error": "ODDT library (openbabel/rdkit) not found on server."}

        try:
            # Load Molecules
            # Auto-detect format based on extension, but PDBQT is standard here
            rec = next(toolkit.readfile('pdbqt', receptor_path))
            lig = next(toolkit.readfile('pdbqt', ligand_path))
            
            # Rec must be explicitly set as protein for some ODDT functions
            rec.protein = True
            
            # 1. Hydrogen Bonds
            # Returns: (mol1_atom_idx, mol2_atom_idx, strict_bool)
            hbs = hbonds(rec, lig)
            num_hbs = len(hbs[0]) if hbs is not None else 0
            
            # 2. Hydrophobic Contacts
            # Returns: (mol1_atom_idx, mol2_atom_idx, distance)
            hyd = hydrophobic_contacts(rec, lig)
            num_hyd = len(hyd[0]) if hyd is not None else 0
            
            # 3. Salt Bridges
            # Returns: (mol1_res_idx, mol2_res_idx, ...)
            slts = salt_bridges(rec, lig)
            num_slts = len(slts) if slts is not None else 0
            
            # 4. Pi-Stacking
            # Returns: (mol1_res_idx, mol2_res_idx, ...)
            pi_s = pi_stacking(rec, lig)
            num_pi = len(pi_s) if pi_s is not None else 0
            
            # Detailed Breakdown (Simplified for JSON)
            details = {
                "hydrogen_bonds": int(num_hbs),
                "hydrophobic_contacts": int(num_hyd),
                "salt_bridges": int(num_slts),
                "pi_stacking": int(num_pi),
                "total_interactions": int(num_hbs + num_hyd + num_slts + num_pi)
            }
            
            logger.info(f"ODDT Analysis: {details}")
            return details
            
        except Exception as e:
            logger.error(f"ODDT Analysis Failed: {e}")
            return {"error": str(e)}

    @staticmethod
    def analyze_from_content(receptor_content: str, ligand_content: str) -> dict:
        """
        Helper to run analysis from string content (saving to temp files).
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as tr:
                tr.write(receptor_content)
                rec_path = tr.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as tl:
                tl.write(ligand_content)
                lig_path = tl.name
                
            result = ODDTAnalysisService.analyze_interactions(rec_path, lig_path)
            
            # Cleanup
            try:
                os.remove(rec_path)
                os.remove(lig_path)
            except: pass
            
            return result
            
        except Exception as e:
            return {"error": f"Temp file handling failed: {e}"}
