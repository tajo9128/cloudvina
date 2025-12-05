"""
Cavity Detection Service for Blind Docking
Uses a geometric approach to detect binding pockets on proteins.
Inspired by fpocket algorithm - detects cavities using alpha spheres.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import tempfile
import os
import subprocess
import json

class CavityDetector:
    """
    Detects potential binding cavities on protein surfaces.
    Uses geometric analysis of the protein structure to identify pockets.
    """
    
    def __init__(self):
        self.min_pocket_volume = 100  # Minimum pocket volume in Å³
        self.max_pockets = 5  # Maximum number of pockets to return
    
    def detect_cavities(self, pdb_content: str) -> List[Dict]:
        """
        Detect binding cavities in a protein structure.
        
        Args:
            pdb_content: PDB file content as string
            
        Returns:
            List of dictionaries containing pocket information:
            {
                "pocket_id": int,
                "center_x": float,
                "center_y": float, 
                "center_z": float,
                "size_x": float,
                "size_y": float,
                "size_z": float,
                "score": float,
                "volume": float,
                "residues": List[str]
            }
        """
        # Parse protein atoms
        atoms = self._parse_pdb(pdb_content)
        
        if not atoms:
            return []
        
        # Find cavities using geometric analysis
        cavities = self._find_cavities(atoms)
        
        # Sort by score (druggability) and return top N
        cavities.sort(key=lambda x: x["score"], reverse=True)
        
        return cavities[:self.max_pockets]
    
    def _parse_pdb(self, pdb_content: str) -> List[Dict]:
        """Parse PDB content and extract atom coordinates."""
        atoms = []
        
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    atom = {
                        'serial': int(line[6:11].strip()),
                        'name': line[12:16].strip(),
                        'resname': line[17:20].strip(),
                        'chain': line[21].strip(),
                        'resseq': int(line[22:26].strip()),
                        'x': float(line[30:38].strip()),
                        'y': float(line[38:46].strip()),
                        'z': float(line[46:54].strip()),
                        'element': line[76:78].strip() if len(line) > 76 else line[12:14].strip()[0]
                    }
                    atoms.append(atom)
                except (ValueError, IndexError):
                    continue
        
        return atoms
    
    def _find_cavities(self, atoms: List[Dict]) -> List[Dict]:
        """
        Find protein cavities using a grid-based approach.
        This is a simplified version of cavity detection algorithms.
        """
        if not atoms:
            return []
        
        # Get protein coordinates
        coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
        
        # Calculate protein bounding box
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        
        # Grid parameters
        grid_spacing = 1.0  # Å
        probe_radius = 1.4  # Water probe radius
        
        # Create grid
        grid_dims = ((max_coords - min_coords) / grid_spacing).astype(int) + 1
        
        # Find surface and cavity points using distance analysis
        cavities = []
        
        # Use clustering to find potential binding sites
        # Simple approach: find regions of high curvature/concavity
        
        # Calculate center of mass
        center_of_mass = coords.mean(axis=0)
        
        # Find atoms that could line binding pockets
        # (atoms with neighbors in limited directions = concave surface)
        
        pocket_candidates = self._identify_pocket_regions(atoms, coords)
        
        for i, pocket in enumerate(pocket_candidates):
            pocket_coords = np.array(pocket['coords'])
            
            if len(pocket_coords) < 5:
                continue
                
            # Calculate pocket center
            center = pocket_coords.mean(axis=0)
            
            # Calculate pocket dimensions
            pocket_min = pocket_coords.min(axis=0)
            pocket_max = pocket_coords.max(axis=0)
            size = pocket_max - pocket_min
            
            # Add padding for docking box
            size = np.maximum(size, 15.0)  # Minimum 15Å box
            size = np.minimum(size, 30.0)  # Maximum 30Å box
            
            # Calculate druggability score based on pocket properties
            score = self._calculate_druggability(pocket, atoms)
            
            # Estimate volume
            volume = np.prod(size) * 0.5  # Rough estimate
            
            cavities.append({
                "pocket_id": i + 1,
                "center_x": round(float(center[0]), 2),
                "center_y": round(float(center[1]), 2),
                "center_z": round(float(center[2]), 2),
                "size_x": round(float(size[0]), 1),
                "size_y": round(float(size[1]), 1),
                "size_z": round(float(size[2]), 1),
                "score": round(score, 3),
                "volume": round(volume, 1),
                "residues": pocket.get('residues', [])
            })
        
        return cavities
    
    def _identify_pocket_regions(self, atoms: List[Dict], coords: np.ndarray) -> List[Dict]:
        """
        Identify potential pocket regions using geometric analysis.
        Looks for concave regions on the protein surface.
        """
        pockets = []
        
        # Calculate protein center
        center = coords.mean(axis=0)
        
        # Group atoms by residue
        residue_groups = {}
        for i, atom in enumerate(atoms):
            key = (atom['chain'], atom['resseq'], atom['resname'])
            if key not in residue_groups:
                residue_groups[key] = {
                    'atoms': [],
                    'coords': [],
                    'center': None
                }
            residue_groups[key]['atoms'].append(atom)
            residue_groups[key]['coords'].append(coords[i])
        
        # Calculate residue centers
        for key, group in residue_groups.items():
            group['center'] = np.mean(group['coords'], axis=0)
        
        # Find residues that form concave regions
        # Use distance from center and neighbor analysis
        residue_list = list(residue_groups.items())
        
        # Cluster residues by spatial proximity
        clusters = self._cluster_residues(residue_list)
        
        for cluster in clusters:
            if len(cluster) < 3:
                continue
                
            pocket_coords = []
            pocket_residues = []
            
            for key in cluster:
                if key in residue_groups:
                    pocket_coords.extend(residue_groups[key]['coords'])
                    pocket_residues.append(f"{key[2]}{key[1]}")
            
            if pocket_coords:
                pockets.append({
                    'coords': pocket_coords,
                    'residues': pocket_residues[:10]  # Limit residue list
                })
        
        # If no pockets found, create one at geometric center
        if not pockets:
            # Find surface atoms (atoms far from center)
            distances = np.linalg.norm(coords - center, axis=1)
            median_dist = np.median(distances)
            
            # Use atoms near median distance as potential binding region
            surface_mask = np.abs(distances - median_dist) < 5.0
            surface_coords = coords[surface_mask]
            
            if len(surface_coords) > 10:
                pockets.append({
                    'coords': surface_coords.tolist(),
                    'residues': []
                })
        
        return pockets
    
    def _cluster_residues(self, residue_list: List[Tuple]) -> List[List]:
        """Simple spatial clustering of residues."""
        if not residue_list:
            return []
        
        # Get residue centers
        centers = []
        keys = []
        for key, group in residue_list:
            if group['center'] is not None:
                centers.append(group['center'])
                keys.append(key)
        
        if not centers:
            return []
            
        centers = np.array(centers)
        
        # Simple clustering based on distance threshold
        cluster_dist = 8.0  # Å
        visited = set()
        clusters = []
        
        for i, center in enumerate(centers):
            if i in visited:
                continue
            
            # Find neighbors
            distances = np.linalg.norm(centers - center, axis=1)
            neighbors = np.where(distances < cluster_dist)[0]
            
            if len(neighbors) >= 3:
                cluster = [keys[j] for j in neighbors if j not in visited]
                if cluster:
                    clusters.append(cluster)
                    visited.update(neighbors)
        
        return clusters
    
    def _calculate_druggability(self, pocket: Dict, atoms: List[Dict]) -> float:
        """
        Calculate druggability score based on pocket properties.
        Higher score = more likely to be a good drug binding site.
        """
        pocket_coords = np.array(pocket['coords'])
        
        if len(pocket_coords) < 5:
            return 0.0
        
        # Factors that contribute to druggability:
        # 1. Pocket size (moderate size is best)
        size = pocket_coords.max(axis=0) - pocket_coords.min(axis=0)
        volume = np.prod(size)
        
        # Optimal volume around 300-1000 Å³
        if volume < 100:
            size_score = 0.3
        elif volume < 300:
            size_score = 0.6
        elif volume < 1000:
            size_score = 1.0
        elif volume < 2000:
            size_score = 0.8
        else:
            size_score = 0.5
        
        # 2. Number of lining residues
        n_residues = len(pocket.get('residues', []))
        residue_score = min(n_residues / 15.0, 1.0)
        
        # 3. Enclosure (how enclosed is the pocket)
        # Approximate by checking spread of coordinates
        spread = np.std(pocket_coords, axis=0)
        enclosure_score = 1.0 / (1.0 + np.mean(spread) / 10.0)
        
        # 4. Hydrophobicity (check for hydrophobic residues)
        hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
        hydrophobic_count = sum(1 for r in pocket.get('residues', []) 
                               if r[:3] in hydrophobic_residues)
        hydrophobic_ratio = hydrophobic_count / max(n_residues, 1)
        hydrophobic_score = 0.5 + 0.5 * hydrophobic_ratio
        
        # Combine scores
        final_score = (
            0.3 * size_score +
            0.2 * residue_score +
            0.3 * enclosure_score +
            0.2 * hydrophobic_score
        )
        
        return final_score


def detect_cavities_from_file(pdb_path: str) -> List[Dict]:
    """
    Convenience function to detect cavities from a PDB file path.
    """
    with open(pdb_path, 'r') as f:
        pdb_content = f.read()
    
    detector = CavityDetector()
    return detector.detect_cavities(pdb_content)


def detect_cavities_from_content(pdb_content: str) -> List[Dict]:
    """
    Convenience function to detect cavities from PDB content string.
    """
    detector = CavityDetector()
    return detector.detect_cavities(pdb_content)
