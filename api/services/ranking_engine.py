
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class RankingEngine:
    """
    Service for prioritizing drug candidates based on consensus scoring.
    Combines Docking Scores, MM-GBSA Binding Energies, and Drug-Likeness (QED).
    """

    def __init__(self, weights: Dict[str, float] = None):
        # Default Weights: prioritize Binding Energy (more accurate) over Docking Score
        self.weights = weights or {
            "docking_score": 0.3,
            "binding_energy": 0.5,
            "qed": 0.2
        }

    def _normalize_score(self, series: pd.Series, lower_is_better: bool = True) -> pd.Series:
        """
        Normalize scores to 0-1 range.
        If lower_is_better (e.g. energies), inversion is applied so 1.0 is the best.
        """
        if series.empty or series.nunique() <= 1:
            return pd.Series([0.5] * len(series), index=series.index)

        min_val = series.min()
        max_val = series.max()
        
        # Avoid division by zero
        if max_val == min_val:
             return pd.Series([0.5] * len(series), index=series.index)

        normalized = (series - min_val) / (max_val - min_val)

        if lower_is_better:
            # Invert: Lowest energy = 1.0, Highest energy = 0.0
            return 1.0 - normalized
        
        return normalized

    def rank_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of hit compounds and append ranking metrics.
        
        Args:
            hits: List of dicts containing 'docking_score', 'binding_energy', 'qin' (QED) etc.
        
        Returns:
            Sorted list of hits with 'consensus_score' and 'rank'.
        """
        if not hits:
            return []

        df = pd.DataFrame(hits)

        # 1. Normalize Docking Score (Lower is better)
        if 'docking_score' in df.columns:
            df['norm_docking'] = self._normalize_score(df['docking_score'].astype(float), lower_is_better=True)
            df['docking_score'] = df['docking_score'].fillna(0) # Handle missing
        else:
            df['norm_docking'] = 0.0

        # 2. Normalize MM-GBSA Energy (Lower is better)
        if 'binding_energy' in df.columns:
            # Handle MM-GBSA failures (NaN)
            df['binding_energy'] = pd.to_numeric(df['binding_energy'], errors='coerce')
            
            # Create mask for valid energies
            valid_mask = df['binding_energy'].notna()
            
            if valid_mask.any():
                df.loc[valid_mask, 'norm_gbsa'] = self._normalize_score(df.loc[valid_mask, 'binding_energy'], lower_is_better=True)
            
            # Penalize missing MM-GBSA values (set to 0 score)
            df['norm_gbsa'] = df['norm_gbsa'].fillna(0.0)
        else:
            df['norm_gbsa'] = 0.0

        # 3. Normalize QED (Higher is better)
        # Assuming we might have simple drug properties
        if 'qed' in df.columns:
             df['norm_qed'] = self._normalize_score(df['qed'].astype(float), lower_is_better=False)
             df['norm_qed'] = df['norm_qed'].fillna(0.0)
        else:
            # If QED missing, use simplified MW penalty proxy or skip
            # For now, just skip
            df['norm_qed'] = 0.5 

        # 4. Calculate Weighted Sum
        df['consensus_score'] = (
            df['norm_docking'] * self.weights['docking_score'] +
            df['norm_gbsa'] * self.weights['binding_energy'] +
            df['norm_qed'] * self.weights['qed']
        )

        # 5. Sort and Rank
        df = df.sort_values(by='consensus_score', ascending=False)
        df['rank'] = range(1, len(df) + 1)

        # Convert back to list of dicts
        # Replace NaN with None for JSON serialization
        return df.replace({np.nan: None}).to_dict(orient='records')
