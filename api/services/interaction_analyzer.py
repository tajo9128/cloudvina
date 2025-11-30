"""
Interaction Analyzer Service
Analyzes protein-ligand interactions using BioPython and geometric distance calculations.
Robust to PDBQT format variations and dependency issues.
"""

import io
import logging
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from Bio.PDB import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Atom import Atom

logger = logging.getLogger(__name__)

class InteractionAnalyzer:
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)

    def analyze_interactions(self, receptor_pdb_content: str, ligand_pdbqt_content: str) -> Dict[str, Any]:
        """
        Analyze interactions between receptor and ligand using geometric criteria.
        
        Args:
            receptor_pdb_content: String content of receptor PDB
            ligand_pdbqt_content: String content of ligand PDBQT (docked pose)
            
        Returns:
            Dictionary containing interaction data
        """
        try:
            # 1. Parse Receptor
            receptor_structure = self.pdb_parser.get_structure("receptor", io.StringIO(receptor_pdb_content))
            atoms = list(receptor_structure.get_atoms())
            ns = NeighborSearch(atoms)
            
            # 2. Parse Ligand PDBQT manually to get coordinates and types
            ligand_atoms = self._parse_pdbqt_atoms(ligand_pdbqt_content)
            
            interactions = {
                "hydrogen_bonds": [],
                "hydrophobic_contacts": [],
                "residues_involved": set()
            }
            
            # 3. Iterate over ligand atoms
            for lig_atom in ligand_atoms:
                lig_name, lig_coord, lig_type = lig_atom
                
                # Search for neighbors within 4.5A
                neighbors = ns.search(lig_coord, 4.5, level='A')
                
                for protein_atom in neighbors:
                    # Calculate exact distance
                    diff = protein_atom.get_coord() - lig_coord
                    dist = float(np.sqrt(np.sum(diff * diff)))
                    
                    # Get residue info
                    residue = protein_atom.get_parent()
                    res_id = f"{residue.get_resname()}{residue.get_id()[1]}"
                    
                    # Check for Hydrogen Bonds (Distance < 3.5A, Polar atoms)
                    if dist < 3.5:
                        if self._is_hbond_candidate(lig_type, protein_atom):
                            interactions["hydrogen_bonds"].append({
                                "ligand_atom": lig_name,
                                "protein_atom": protein_atom.get_name(),
                                "residue": res_id,
                                "distance": round(dist, 2)
                            })
                            interactions["residues_involved"].add(res_id)
                            continue 
                            
                    # Check for Hydrophobic Contacts (Distance < 4.5A, C-C)
                    if dist < 4.5:
                        if self._is_hydrophobic_candidate(lig_type, protein_atom):
                            interactions["hydrophobic_contacts"].append({
                                "ligand_atom": lig_name,
                                "protein_atom": protein_atom.get_name(),
                                "residue": res_id,
                                "distance": round(dist, 2)
                            })
                            interactions["residues_involved"].add(res_id)

            # Format results
            interactions["residues_involved"] = list(interactions["residues_involved"])
            interactions["hydrogen_bonds"] = self._deduplicate_by_residue(interactions["hydrogen_bonds"])
            interactions["hydrophobic_contacts"] = self._deduplicate_by_residue(interactions["hydrophobic_contacts"])
            
            return interactions

        except Exception as e:
            logger.error(f"Error analyzing interactions: {str(e)}")
            # Return empty structure on error to avoid API failure
            return {
                "hydrogen_bonds": [],
                "hydrophobic_contacts": [],
                "residues_involved": []
            }

    def _parse_pdbqt_atoms(self, pdbqt_content: str) -> List[Tuple[str, np.ndarray, str]]:
        """
        Extract atom coordinates and types from PDBQT content.
        Returns list of (atom_name, coordinates, atom_type).
        """
        atoms = []
        for line in pdbqt_content.splitlines():
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    # PDBQT format is similar to PDB
                    # Name: cols 12-16, X: 30-38, Y: 38-46, Z: 46-54
                    # Type is usually at the end
                    name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    # PDBQT atom type is often the last column or derived from name
                    # Vina PDBQT usually has type at the end (e.g., "OA", "C", "HD")
                    parts = line.split()
                    atom_type = parts[-1] if len(parts) > 0 else name[0]
                    
                    atoms.append((name, np.array([x, y, z], dtype="f"), atom_type))
                except (ValueError, IndexError):
                    continue
        return atoms

    def _is_hbond_candidate(self, lig_type: str, protein_atom: Atom) -> bool:
        """Check if pair can form Hydrogen Bond."""
        polar_types = {'N', 'O', 'F', 'S', 'OA', 'NA', 'SA', 'HD'}
        
        # Clean up types (remove numbers, etc.)
        lig_t = ''.join([c for c in lig_type if c.isalpha()])
        prot_t = protein_atom.element
        
        return lig_t in polar_types and prot_t in polar_types

    def _is_hydrophobic_candidate(self, lig_type: str, protein_atom: Atom) -> bool:
        """Check if pair is hydrophobic (C-C)."""
        lig_t = ''.join([c for c in lig_type if c.isalpha()])
        prot_t = protein_atom.element
        
        # PDBQT Carbon types: C, A (aromatic), etc.
        carbon_types = {'C', 'A'}
        
        return lig_t in carbon_types and prot_t == 'C'

    def _deduplicate_by_residue(self, interaction_list: List[Dict]) -> List[Dict]:
        """Keep only the closest interaction per residue."""
        best_per_residue = {}
        for interaction in interaction_list:
            res = interaction['residue']
            dist = interaction['distance']
            if res not in best_per_residue or dist < best_per_residue[res]['distance']:
                best_per_residue[res] = interaction
        return list(best_per_residue.values())
