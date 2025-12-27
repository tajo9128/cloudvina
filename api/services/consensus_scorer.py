"""
Weighted Consensus Scoring Service (Step 5)
Combines Physics (Vina), Deep Learning (Gnina), and ML (RF) into a single Metascore.
"""
import logging

logger = logging.getLogger("consensus_scorer")

class ConsensusScorer:
    """
    Implements the Percentile-Based Consensus Scoring (BioDockify-Integrated).
    International Standard:
    Consensus = Mean(Percentile(Vina), Percentile(Gnina), Percentile(RF))
    Confidence = 1.0 - StdDev(Percentiles)
    """
    
    import statistics

    @staticmethod
    def calculate_percentile(value, ref_min, ref_max):
        """
        Maps a raw score to a 0.0-1.0 percentile rank based on reference range.
        """
        if value is None: return 0.0
        
        try:
            val = float(value)
            
            # Logic: We want to position 'val' between 'start' (0.0) and 'end' (1.0)
            start = ref_min # 0.0 point
            end = ref_max   # 1.0 point
            
            # Determine direction (Ascending or Descending?)
            # If end > start (e.g. RF: 4.0 -> 8.0), normal map.
            # If start > end (e.g. Vina: -4.0 -> -12.0), inverted map.
            
            # Simple Linear Interpolation: (val - start) / (end - start)
            # Example Vina: val=-8. (-8 - (-4)) / (-12 - (-4)) = -4 / -8 = 0.5. Correct.
            # Example Vina: val=-13 (Better than elite). (-13 - (-4)) / -8 = -9/-8 = 1.125 -> Clamp to 1.0.
            # Example Vina: val=-2 (Worse than bad). (-2 - (-4)) / -8 = 2/-8 = -0.25 -> Clamp to 0.0.
            
            raw_pct = (val - start) / (end - start)
            return max(0.0, min(1.0, raw_pct))
                
        except:
            return 0.0

    @staticmethod
    def calculate_score(vina_score, cnn_score, rf_score=None):
        """
        Compute the Tri-Score Consensus.
        Returns:
            dict with 'consensus_score' (0-10), 'agreement_confidence' (0-1), 'percentiles'
        """
        try:
            # 1. Calculate Percentiles (Reference: BioDockify Integrated Solution)
            # Vina: -4.0 (0%) -> -12.0 (100%)
            p_vina = ConsensusScorer.calculate_percentile(vina_score, -4.0, -12.0)
            
            # Gnina: 0.1 (0%) -> 0.9 (100%)
            p_cnn = ConsensusScorer.calculate_percentile(cnn_score, 0.1, 0.9)
            
            # RF: 4.0 (0%) -> 8.0 (100%)
            if rf_score is not None:
                p_rf = ConsensusScorer.calculate_percentile(rf_score, 4.0, 8.0)
                scores = [p_vina, p_cnn, p_rf]
            else:
                p_rf = 0.0
                scores = [p_vina, p_cnn]
            
            # 2. Consensus Score (Mean of Percentiles)
            import statistics
            consensus_pct = statistics.mean(scores)
            
            # 3. Agreement Confidence (1 - StdDev)
            if len(scores) > 1:
                stdev = statistics.stdev(scores)
                # Cap confidence at 0
                confidence = max(0.0, 1.0 - stdev)
            else:
                confidence = 0.0 
                
            return {
                "consensus_score": round(consensus_pct * 10, 2), # Scale 0-10 for UI
                "agreement_confidence": round(confidence, 2),
                "percentiles": {
                    "vina": round(p_vina, 2),
                    "cnn": round(p_cnn, 2),
                    "rf": round(p_rf, 2) if rf_score is not None else 0.0
                }
            }

        except Exception as e:
            logger.error(f"Consensus Calc Failed: {e}")
            return {"consensus_score": 0.0, "agreement_confidence": 0.0}
            
    @staticmethod
    def get_confidence_label(consensus_score, agreement=None):
        """Returns label based on Score AND Agreement"""
        # If agreement is provided, use it for stricter labeling
        if agreement is not None and agreement < 0.6:
            return "Disagreement (Verify)"
            
        if consensus_score >= 8.0: return "Elite Binder"
        if consensus_score >= 6.5: return "Strong Binder"
        if consensus_score >= 5.0: return "Moderate Binder"
        return "Weak Binder"
