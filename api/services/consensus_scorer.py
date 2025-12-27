"""
Weighted Consensus Scoring Service (Step 5)
Combines Physics (Vina), Deep Learning (Gnina), and ML (RF) into a single Metascore.
"""
import logging

logger = logging.getLogger("consensus_scorer")

class ConsensusScorer:
    """
    Implements the Weighted Consensus Formula.
    International Standard:
    Consensus = 0.4*Vina + 0.4*Gnina + 0.2*RF
    """
    
    @staticmethod
    def calculate_score(vina_score, cnn_score, rf_score=None):
        """
        Compute the weighted consensus score.
        Inputs:
            vina_score: kcal/mol (usually negative, e.g. -9.0)
            cnn_score: 0.0-1.0 (probability)
            rf_score: pKd (e.g. 7.0) or None if not available
        
        Returns:
            Review Score (0-10 scale, higher is better)
        """
        try:
            # 1. Normalize Vina (-12 to -4 -> 0 to 1)
            # We want more negative to be higher score.
            # Map -12 -> 1.0, -4 -> 0.0
            vina_val = float(vina_score) if vina_score is not None else 0
            # Clamp between -13 and -4 for normalization
            v_clamped = max(-13, min(vina_val, -4))
            # Formula: (Val - Min) / (Max - Min) inverted
            # (-9 - (-4)) / (-13 - (-4)) = -5 / -9 = 0.55
            norm_vina = (v_clamped - (-4)) / (-13 - (-4))
            
            # 2. Normalize Gnina (already 0-1)
            cnn_val = float(cnn_score) if cnn_score is not None else 0
            norm_cnn = cnn_val
            
            # 3. Normalize RF (pKd 4-10 -> 0-1)
            # pKd 9 = nanomolar, pKd 4 = millimolar
            rf_val = float(rf_score) if rf_score is not None else 6.0 # Default to moderate
            r_clamped = max(4, min(rf_val, 10))
            norm_rf = (r_clamped - 4) / (10 - 4)
            
            # 4. Weighted Sum
            # If RF is missing (2-score consensus), redistribute weight
            if rf_score is None:
                # 0.5 Vina + 0.5 Gnina
                final_score = (0.5 * norm_vina) + (0.5 * norm_cnn)
            else:
                # 0.4 Vina + 0.4 Gnina + 0.2 RF
                final_score = (0.4 * norm_vina) + (0.4 * norm_cnn) + (0.2 * norm_rf)
            
            # Scale to 0-10 for user display
            display_score = round(final_score * 10, 2)
            
            return display_score
            
        except Exception as e:
            logger.error(f"Consensus Calculation Failed: {e}")
            return 0.0
            
    @staticmethod
    def get_confidence_label(consensus_score):
        """Returns a string label for the score (0-10)."""
        if consensus_score >= 8.0: return "Elite Binder"
        if consensus_score >= 6.5: return "Strong Binder"
        if consensus_score >= 5.0: return "Moderate Binder"
        return "Weak Binder"
