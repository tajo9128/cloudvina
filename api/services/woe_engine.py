class WeightOfEvidenceEngine:
    """
    BioDockify NAM Weight-of-Evidence (WoE) Engine (v1.0)
    Integrates Mechanistic, Dynamic, and Safety evidence into a single decision score.
    """
    
    # Weights defined in Blueprint
    WEIGHTS = {
        "DOCKING": 0.30,
        "MD": 0.30,
        "TOX_ADMET": 0.40
    }

    def calculate_confidence(self, docking_data, md_data, tox_data):
        """
        Calculates the Integrated NAM Confidence Score (0-100%).
        
        Args:
            docking_data (dict): {'affinity': float, 'consensus_agreement': float (0-1)}
            md_data (dict): {'rmsd_stable': bool, 'contact_persistence': float (0-1)}
            tox_data (dict): {'flags': int, 'model_agreement': float (0-1)}
        
        Returns:
            dict: {
                'total_score': float,
                'tier': str (HIGH/MED/LOW),
                'breakdown': dict
            }
        """
        
        # 1. Docking Score Component (30%)
        # Logic: High affinity (< -7.0) AND High Consensus Agreement -> High Score
        dock_raw = 0
        if docking_data.get('affinity', 0) < -7.0:
            dock_raw += 50
        dock_raw += (docking_data.get('consensus_agreement', 0) * 50) # Add up to 50 based on agreement
        
        dock_weighted = dock_raw * self.WEIGHTS["DOCKING"]

        # 2. MD Stability Component (30%)
        # Logic: RMSD < 3A is stable.
        md_raw = 0
        if md_data.get('rmsd_stable', False):
            md_raw = 100 # Stable = 100
        else:
            md_raw = 20 # Unstable = minimal points
        
        md_weighted = md_raw * self.WEIGHTS["MD"]

        # 3. Tox / ADMET Component (40%)
        # Logic: Start at 100, subtract for flags
        tox_raw = 100
        flags = tox_data.get('flags', 0)
        tox_raw -= (flags * 25) # Each flag drops score by 25
        if tox_raw < 0: tox_raw = 0
        
        tox_weighted = tox_raw * self.WEIGHTS["TOX_ADMET"]

        # Final Sum
        total_score = dock_weighted + md_weighted + tox_weighted
        
        # Determine Tier
        tier = "LOW"
        if total_score >= 80:
            tier = "HIGH"
        elif total_score >= 50:
            tier = "MEDIUM"

        return {
            "total_score": round(total_score, 1),
            "tier": tier,
            "breakdown": {
                "docking_contribution": round(dock_weighted, 1),
                "md_contribution": round(md_weighted, 1),
                "tox_contribution": round(tox_weighted, 1)
            }
        }
