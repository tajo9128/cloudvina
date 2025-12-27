"""
Layer 1: Protein Flexibility Generation
Generates ensemble of [Crystal, AlphaFold, NMA]
Does NOT modify existing docking pipeline
"""

import logging
import os
from typing import List, Optional
from utils.alphafold_fetch import AlphaFoldFetcher
from utils.nma import generate_nma_conformers

logger = logging.getLogger("layer1_generator")

class Layer1Generator:
    """Generate protein ensemble for flexible docking"""
    
    def __init__(self, pdb_file: str, job_id: str, use_af: bool = True, use_nma: bool = True):
        self.pdb_file = pdb_file
        self.job_id = job_id
        self.use_af = use_af
        self.use_nma = use_nma
        self.ensemble_pdbs = []
    
    def generate(self) -> List[str]:
        """
        Generate ensemble structures
        
        Returns:
            List of PDB file paths (crystal always first)
            
        Example:
            ["crystal.pdb", "AF_aligned.pdb", "nma_mode1.pdb"]
        """
        
        # ALWAYS: Add crystal structure
        self.ensemble_pdbs.append(self.pdb_file)
        logger.info(f"[{self.job_id}] ✓ Layer 1: Crystal structure added: {self.pdb_file}")
        
        # OPTIONAL: Add AlphaFold
        if self.use_af:
            af_pdb = self._add_alphafold()
            if af_pdb:
                self.ensemble_pdbs.append(af_pdb)
        
        # OPTIONAL: Add NMA conformer
        if self.use_nma:
            nma_pdb = self._add_nma()
            if nma_pdb:
                self.ensemble_pdbs.append(nma_pdb)
        
        logger.info(f"[{self.job_id}] Layer 1 Complete: {len(self.ensemble_pdbs)} structures ready.")
        return self.ensemble_pdbs
    
    def _add_alphafold(self) -> Optional[str]:
        """Fetch AlphaFold (isolated, non-intrusive)"""
        try:
            logger.info(f"[{self.job_id}] Layer 1: Fetching AlphaFold...")
            
            fetcher = AlphaFoldFetcher()
            # Assuming pdb_file name contains PDB ID or UniProt, but for now mostly relying on filename
            # In a real scenario, we'd parse the file header or filename.
            # Simplified: Try to parse PDB ID from filename (e.g. 1hgs.pdb)
            filename = os.path.basename(self.pdb_file)
            pdb_id = filename.split('.')[0].upper()
            
            if len(pdb_id) != 4:
                # If local file doesn't look like PDB ID, we might skip AF
                # or rely on header parsing. For safety, we skip.
                logger.warning(f"[{self.job_id}] Filename '{filename}' not a valid PDB ID. Skipping AlphaFold.")
                return None

            result = fetcher.fetch_by_pdb_id(pdb_id, output_dir=os.path.dirname(self.pdb_file))
            
            if result:
                logger.info(f"[{self.job_id}] ✓ AlphaFold fetched: {result}")
                return result
            else:
                logger.warning(f"[{self.job_id}] AlphaFold fetch returned nothing.")
                return None
        
        except Exception as e:
            logger.warning(f"[{self.job_id}] AlphaFold step failed: {e}")
            return None
    
    def _add_nma(self) -> Optional[str]:
        """Generate NMA conformer (isolated, non-intrusive)"""
        try:
            logger.info(f"[{self.job_id}] Layer 1: Generating NMA conformer...")
            
            # Simple perturbation
            output_path = self.pdb_file.replace(".pdb", "_nma.pdb")
            
            nma_pdb = generate_nma_conformers(
                input_pdb=self.pdb_file,
                output_pdb=output_path,
                mode=1,
                rmsd=2.0 # 2 Angstrom perturbation
            )
            
            if nma_pdb:
                logger.info(f"[{self.job_id}] ✓ NMA conformer generated: {nma_pdb}")
                return nma_pdb
            else:
                logger.warning(f"[{self.job_id}] NMA generation failed")
                return None
        
        except Exception as e:
            logger.warning(f"[{self.job_id}] NMA step failed: {e}")
            return None
