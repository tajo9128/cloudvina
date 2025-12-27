
"""
RF Score-based filtering tiers
Determines which analyses to run based on confidence levels.
"""
import os

class ScoringTiers:
    """
    Tier definitions based on RF model calibration.
    Controls the "Smart Funnel" logic.
    """
    
    # Tier boundaries (Configurable via Env)
    TIER_1_HIGH = float(os.getenv('BIODOCKIFY_TIER1_HIGH', 6.5))         # High confidence binder
    TIER_2_MEDIUM_HIGH = float(os.getenv('BIODOCKIFY_TIER2_HIGH', 5.5))  # Likely binder
    TIER_2_MEDIUM_LOW = float(os.getenv('BIODOCKIFY_TIER2_LOW', 4.5))    # Possible binder
    TIER_3_LOW = float(os.getenv('BIODOCKIFY_TIER3', 3.5))               # Unlikely binder
    
    # Vina-RF anomaly detection
    # If Vina says "Great" (-9.0) but RF says "Terrible" (4.0), it's an anomaly.
    VINA_STRONG_BINDER = -8.0
    RF_WEAK_LIMIT = 4.5
    
    @classmethod
    def get_tier(cls, rf_score: float, vina_affinity: float = None) -> dict:
        """
        Classify a compound into tier based on RF score
        Also check for anomalies (high Vina, low RF)
        
        Returns:
            {
                'tier': 'Tier1' | 'Tier2' | 'Tier3',
                'label': str,
                'gnina_enabled': bool,
                'minimization_enabled': bool,
                'md_enabled': bool,
                'is_anomaly': bool,
                'reason': str
            }
        """
        
        # Check for Vina-RF anomaly
        is_anomaly = False
        anomaly_reason = ""
        
        if vina_affinity is not None:
            # If Vina shows strong binding but RF disagrees, flag it
            if vina_affinity < cls.VINA_STRONG_BINDER and rf_score < cls.RF_WEAK_LIMIT:
                is_anomaly = True
                anomaly_reason = "High Vina affinity vs Low RF score - potential scaffold novelty"
        
        # Tier assignment
        if rf_score >= cls.TIER_1_HIGH:
            tier_info = {
                'tier': 'Tier1',
                'label': 'High Confidence Binder',
                'gnina_enabled': True,
                'minimization_enabled': True,
                'is_anomaly': is_anomaly,
                'reason': 'Strong RF prediction'
            }
        
        elif rf_score >= cls.TIER_2_MEDIUM_HIGH:
            tier_info = {
                'tier': 'Tier2_High',
                'label': 'Medium-High Confidence',
                'gnina_enabled': True,
                'minimization_enabled': True,
                'is_anomaly': is_anomaly,
                'reason': 'Moderate RF prediction'
            }
        
        elif rf_score >= cls.TIER_2_MEDIUM_LOW:
            tier_info = {
                'tier': 'Tier2_Low',
                'label': 'Medium-Low Confidence',
                'gnina_enabled': True,
                'minimization_enabled': False,  # Skip minimization to save time
                'is_anomaly': is_anomaly,
                'reason': 'Weak RF prediction'
            }
        
        elif rf_score >= cls.TIER_3_LOW:
            tier_info = {
                'tier': 'Tier3',
                'label': 'Low Confidence (Filtered)',
                'gnina_enabled': False,  # Skip GNINA (expensive)
                'minimization_enabled': False,
                'is_anomaly': is_anomaly,
                'reason': 'Poor RF prediction'
            }
        
        else:
            tier_info = {
                'tier': 'Tier3_Very_Low',
                'label': 'Very Low Confidence (Filtered)',
                'gnina_enabled': False,
                'minimization_enabled': False,
                'is_anomaly': is_anomaly,
                'reason': 'Very poor RF prediction'
            }
        
        # Override: If anomaly detected, enable GNINA & minimization
        if is_anomaly:
            tier_info['gnina_enabled'] = True
            tier_info['minimization_enabled'] = True
            tier_info['is_anomaly'] = True
            tier_info['tier'] = f"{tier_info['tier']} (Anomaly)"
            tier_info['anomaly_reason'] = anomaly_reason
        
        return tier_info
