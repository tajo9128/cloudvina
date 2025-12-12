from typing import Dict, List, Optional
import os

class RDockParser:
    """
    Parses rDock output (.sd file) to extract scores and poses.
    rDock scores are stored in SDF tags: <SCORE.INTER>
    """
    
    def __init__(self, sdf_path: str):
        self.sdf_path = sdf_path
        self.poses = []
        self.best_score = None
    
    def parse(self) -> Dict:
        if not os.path.exists(self.sdf_path):
            return {}
            
        with open(self.sdf_path, 'r') as f:
            content = f.read()
            
        # Split by molecule delimiter ($$$$)
        molecules = content.split('$$$$')
        
        parsed_poses = []
        
        for i, mol in enumerate(molecules):
            if not mol.strip():
                continue
                
            # Extract SCORE.INTER
            # Format:
            # >  <SCORE.INTER>
            # -24.532
            
            try:
                score_block = mol.split('<SCORE.INTER>')
                if len(score_block) > 1:
                    score_line = score_block[1].split('\n')[1].strip()
                    score = float(score_line)
                    
                    parsed_poses.append({
                        "mode": i + 1,
                        "affinity": score, # rDock score (arbitrary units)
                        "score_type": "rDock (SCORE.INTER)",
                        "rmsd_lb": 0.0, # rDock doesn't output RMSD vs input automatically
                        "rmsd_ub": 0.0
                    })
            except Exception as e:
                print(f"Error parsing molecule {i}: {e}")
                continue
        
        # Sort by best score (most negative is better)
        parsed_poses.sort(key=lambda x: x['affinity'])
        
        self.poses = parsed_poses
        if parsed_poses:
            self.best_score = parsed_poses[0]['affinity']
            
        return {
            "poses": self.poses,
            "best_affinity": self.best_score,
            "num_poses": len(self.poses),
            "scoring_function": "rDock"
        }

def parse_rdock_output(sdf_path: str) -> Dict:
    parser = RDockParser(sdf_path)
    return parser.parse()
