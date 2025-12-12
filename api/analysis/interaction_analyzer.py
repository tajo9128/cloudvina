"""
Protein-Ligand Interaction Profiler (PLIP) integration
Identifies hydrogen bonds, hydrophobic contacts, salt bridges, etc.
"""

from plip.structure.preparation import PDBComplex
import pandas as pd
from typing import Dict, List, Any


class InteractionAnalyzer:
    """Analyze protein-ligand interactions using PLIP"""
    
    def __init__(self, pdb_file: str, ligand_identifier: str = 'LIG'):
        """
        Args:
            pdb_file: PDB file of protein-ligand complex
            ligand_identifier: Residue name of ligand (e.g., 'LIG')
        """
        self.pdb_file = pdb_file
        self.ligand_identifier = ligand_identifier
    
    def analyze_interactions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify all interactions in a single frame
        
        Returns:
            Dict with interaction types and their properties
        """
        my_mol = PDBComplex()
        my_mol.load_pdb(self.pdb_file)
        
        # Run interaction analysis
        my_mol.analyze()
        
        interactions = {
            'hydrogen_bonds': [],
            'hydrophobic_contacts': [],
            'pi_stacking': [],
            'salt_bridges': [],
            'halogen_bonds': [],
            'water_bridges': []
        }
        
        # Extract interactions from binding site
        for bs in my_mol.interaction_sets:
            if bs.lig_resname == self.ligand_identifier:
                # Hydrogen bonds
                for hb in bs.hbonds:
                    interactions['hydrogen_bonds'].append({
                        'donor': f"{hb.don_chain}:{hb.don_idx}",
                        'acceptor': f"{hb.acc_chain}:{hb.acc_idx}",
                        'distance': hb.distance_h_a,
                        'angle': hb.angle
                    })
                
                # Hydrophobic contacts
                for hc in bs.hydrophobic_contacts:
                    interactions['hydrophobic_contacts'].append({
                        'atom1': f"{hc.atom1.chain}:{hc.atom1.resnr}",
                        'atom2': f"{hc.atom2.chain}:{hc.atom2.resnr}",
                        'distance': hc.distance
                    })
                
                # Pi-stacking
                for ps in bs.pistacking:
                    interactions['pi_stacking'].append({
                        'ring1': f"{ps.ring1_atom1.chain}:{ps.ring1_atom1.resnr}",
                        'ring2': f"{ps.ring2_atom1.chain}:{ps.ring2_atom1.resnr}",
                        'distance': ps.distance,
                        'angle': ps.angle
                    })
                
                # Salt bridges
                for sb in bs.saltbridges:
                    interactions['salt_bridges'].append({
                        'cation': f"{sb.positive_reschain}:{sb.positive_resnum}",
                        'anion': f"{sb.negative_reschain}:{sb.negative_resnum}",
                        'distance': sb.distance
                    })
                
                # Halogen bonds
                for xb in bs.halogen_bonds:
                    interactions['halogen_bonds'].append({
                        'donor': f"{xb.don.chain}:{xb.don.resnr}",
                        'acceptor': f"{xb.acc.chain}:{xb.acc.resnr}",
                        'distance': xb.distance,
                        'angle': xb.angle
                    })
                
                # Water bridges
                for wb in bs.water_bridges:
                    interactions['water_bridges'].append({
                        'donor': f"{wb.don.chain}:{wb.don.resnr}",
                        'acceptor': f"{wb.acc.chain}:{wb.acc.resnr}",
                        'water': f"{wb.water.chain}:{wb.water.resnr}",
                        'distance_donor': wb.distance_aw,
                        'distance_acceptor': wb.distance_dw
                    })
        
        return interactions
    
    def create_interaction_fingerprint(self, interactions: Dict[str, List]) -> Dict[str, Any]:
        """
        Create a binary fingerprint of interactions
        Useful for comparison across trajectory frames
        
        Args:
            interactions: Output from analyze_interactions()
            
        Returns:
            Fingerprint dict with counts and key residues
        """
        fingerprint = {}
        
        for interaction_type, interaction_list in interactions.items():
            fingerprint[f"{interaction_type}_count"] = len(interaction_list)
            
            # Most frequent interacting residues
            if interaction_list:
                if interaction_type in ['hydrogen_bonds', 'hydrophobic_contacts']:
                    partners = [i.get('donor', i.get('atom1', '')) 
                               for i in interaction_list]
                    fingerprint[f"{interaction_type}_residues"] = list(set(partners))
        
        return fingerprint
    
    def get_interaction_summary(self, interactions: Dict[str, List]) -> pd.DataFrame:
        """
        Convert interactions to summary DataFrame
        
        Args:
            interactions: Output from analyze_interactions()
            
        Returns:
            DataFrame with interaction counts
        """
        summary = {
            'interaction_type': [],
            'count': []
        }
        
        for interaction_type, interaction_list in interactions.items():
            summary['interaction_type'].append(interaction_type)
            summary['count'].append(len(interaction_list))
        
        return pd.DataFrame(summary)
    
    def export_interactions(self, interactions: Dict[str, List], output_file: str):
        """
        Export interactions to CSV
        
        Args:
            interactions: Output from analyze_interactions()
            output_file: Path to save CSV
        """
        all_interactions = []
        
        for interaction_type, interaction_list in interactions.items():
            for interaction in interaction_list:
                row = {'type': interaction_type}
                row.update(interaction)
                all_interactions.append(row)
        
        df = pd.DataFrame(all_interactions)
        df.to_csv(output_file, index=False)
        
        return output_file
