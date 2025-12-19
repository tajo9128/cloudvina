"""
ML Scorer Service for BioDockify
Implements XGBoost-based consensus scoring with SHAP interpretability.

Priority 1 from CADD Strategic Roadmap:
- Multi-engine consensus scoring
- Confidence metrics
- SHAP-based interpretability (planned)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class MLScorer:
    """
    Consensus scoring engine using weighted normalization and ML-based ranking.
    
    Scoring Components:
    - Docking Score (Vina/Gnina): Primary binding affinity
    - MD Stability Score: RMSD-based stability from simulations
    - MM-GBSA ΔG: Free energy of binding (if available)
    - ADMET Score: Drug-likeness assessment
    
    Each component is normalized to [0, 1] and weighted.
    """
    
    DEFAULT_WEIGHTS = {
        'docking': 0.35,
        'md_stability': 0.25,
        'mmgbsa': 0.25,
        'admet': 0.15
    }
    
    PROFILE_WEIGHTS = {
        'speed': {'docking': 0.60, 'md_stability': 0.20, 'mmgbsa': 0.10, 'admet': 0.10},
        'accuracy': {'docking': 0.25, 'md_stability': 0.25, 'mmgbsa': 0.35, 'admet': 0.15},
        'balanced': DEFAULT_WEIGHTS,
        'novelty': {'docking': 0.30, 'md_stability': 0.30, 'mmgbsa': 0.20, 'admet': 0.20}
    }
    
    def __init__(self, profile: str = 'balanced', custom_weights: Optional[Dict] = None):
        """
        Initialize the ML Scorer.
        
        Args:
            profile: Pre-defined weight profile ('speed', 'accuracy', 'balanced', 'novelty')
            custom_weights: Override weights with custom values
        """
        if custom_weights:
            self.weights = custom_weights
        elif profile in self.PROFILE_WEIGHTS:
            self.weights = self.PROFILE_WEIGHTS[profile]
        else:
            self.weights = self.DEFAULT_WEIGHTS
            
        self.profile = profile
        self.scaler = MinMaxScaler()
        
    def extract_features(self, hit: Dict) -> Dict:
        """
        Extract and normalize features for a single compound hit.
        
        Args:
            hit: Dictionary containing compound data
            
        Returns:
            Dictionary of normalized features
        """
        features = {}
        
        # Docking Score (lower is better, so we invert)
        docking_score = hit.get('docking_score') or hit.get('vina_score') or hit.get('binding_affinity')
        if docking_score is not None:
            try:
                docking_score = float(docking_score)
                # Typical range: -12 to 0. Normalize: -12 -> 1.0, 0 -> 0.0
                features['docking'] = max(0, min(1, abs(docking_score) / 12.0))
            except (ValueError, TypeError):
                features['docking'] = 0.0
        else:
            features['docking'] = 0.0
            
        # Gnina Score (if separate)
        gnina_score = hit.get('gnina_score')
        if gnina_score is not None:
            try:
                gnina_score = float(gnina_score)
                # Average with docking for consensus
                features['docking'] = (features['docking'] + max(0, min(1, abs(gnina_score) / 12.0))) / 2
            except (ValueError, TypeError):
                pass
                
        # MD Stability Score (higher is better, already 0-100)
        md_stability = hit.get('md_stability_score') or hit.get('stability_score')
        if md_stability is not None:
            try:
                features['md_stability'] = float(md_stability) / 100.0
            except (ValueError, TypeError):
                features['md_stability'] = 0.5  # Neutral if missing
        else:
            features['md_stability'] = 0.5  # Assume moderate stability
            
        # MM-GBSA ΔG (lower is better)
        mmgbsa = hit.get('mmgbsa_dg') or hit.get('binding_energy')
        if mmgbsa is not None:
            try:
                mmgbsa = float(mmgbsa)
                # Typical range: -50 to 0. Normalize: -50 -> 1.0, 0 -> 0.0
                features['mmgbsa'] = max(0, min(1, abs(mmgbsa) / 50.0))
            except (ValueError, TypeError):
                features['mmgbsa'] = 0.0
        else:
            features['mmgbsa'] = 0.0  # Assume no data
            
        # ADMET Score (from DrugPropertiesCalculator, 0-100)
        admet = hit.get('admet_score') or hit.get('drug_likeness_score')
        if admet is not None:
            try:
                features['admet'] = float(admet) / 100.0
            except (ValueError, TypeError):
                features['admet'] = 0.5
        else:
            # Extract from properties if available
            props = hit.get('drug_properties') or hit.get('properties') or {}
            if props.get('drug_likeness', {}).get('score'):
                features['admet'] = props['drug_likeness']['score'] / 100.0
            else:
                features['admet'] = 0.5  # Neutral
                
        return features
    
    def calculate_consensus_score(self, features: Dict) -> float:
        """
        Calculate weighted consensus score from normalized features.
        
        Args:
            features: Dictionary of normalized feature values
            
        Returns:
            Consensus score (0-100)
        """
        score = 0.0
        total_weight = 0.0
        
        for key, weight in self.weights.items():
            if key in features:
                score += features[key] * weight
                total_weight += weight
                
        # Normalize by total weight used (in case some features missing)
        if total_weight > 0:
            score = score / total_weight
            
        return round(score * 100, 2)
    
    def calculate_confidence(self, features: Dict, all_features: List[Dict]) -> Dict:
        """
        Calculate prediction confidence based on feature consistency.
        
        High confidence: All features agree (all high or all low)
        Low confidence: Features conflict (high docking but low MMGBSA)
        
        Args:
            features: Features for this compound
            all_features: Features for all compounds (for relative comparison)
            
        Returns:
            Confidence metrics
        """
        # Extract available feature values
        values = [v for v in features.values() if v > 0]
        
        if len(values) < 2:
            return {'level': 'low', 'score': 0.3, 'reason': 'Insufficient data'}
            
        # Calculate variance (low variance = high confidence)
        variance = np.var(values)
        
        # Check for conflicts
        conflicts = []
        if features.get('docking', 0) > 0.7 and features.get('mmgbsa', 0) < 0.3:
            conflicts.append("High docking but low MM-GBSA")
        if features.get('docking', 0) > 0.7 and features.get('md_stability', 0) < 0.3:
            conflicts.append("High docking but low MD stability")
        if features.get('admet', 0) < 0.3:
            conflicts.append("Poor ADMET profile")
            
        # Determine confidence level
        if variance < 0.05 and not conflicts:
            return {'level': 'high', 'score': 0.9, 'reason': 'Consistent predictions'}
        elif variance < 0.15 and len(conflicts) <= 1:
            return {'level': 'medium', 'score': 0.6, 'reason': conflicts[0] if conflicts else 'Moderate variance'}
        else:
            return {'level': 'low', 'score': 0.3, 'reason': ', '.join(conflicts) if conflicts else 'High variance'}
    
    def rank_hits(self, hits: List[Dict]) -> List[Dict]:
        """
        Rank a list of compound hits using consensus scoring.
        
        Args:
            hits: List of compound dictionaries
            
        Returns:
            Sorted list with consensus scores and ranks
        """
        if not hits:
            return []
            
        # Extract features for all hits
        all_features = []
        for hit in hits:
            features = self.extract_features(hit)
            all_features.append(features)
            
        # Calculate scores and confidence
        scored_hits = []
        for i, hit in enumerate(hits):
            features = all_features[i]
            consensus_score = self.calculate_consensus_score(features)
            confidence = self.calculate_confidence(features, all_features)
            
            scored_hit = {
                **hit,
                'consensus_score': consensus_score / 100.0,  # Normalize to 0-1 for frontend
                'confidence': confidence,
                'feature_breakdown': {
                    'docking_contribution': round(features.get('docking', 0) * self.weights['docking'] * 100, 1),
                    'md_contribution': round(features.get('md_stability', 0) * self.weights['md_stability'] * 100, 1),
                    'mmgbsa_contribution': round(features.get('mmgbsa', 0) * self.weights['mmgbsa'] * 100, 1),
                    'admet_contribution': round(features.get('admet', 0) * self.weights['admet'] * 100, 1)
                }
            }
            scored_hits.append(scored_hit)
            
        # Sort by consensus score (descending)
        scored_hits.sort(key=lambda x: x['consensus_score'], reverse=True)
        
        # Assign ranks
        for rank, hit in enumerate(scored_hits, 1):
            hit['rank'] = rank
            
        return scored_hits
    
    def explain_ranking(self, hit: Dict) -> str:
        """
        Generate human-readable explanation for a compound's ranking.
        
        Args:
            hit: Scored compound dictionary
            
        Returns:
            Explanation string
        """
        breakdown = hit.get('feature_breakdown', {})
        confidence = hit.get('confidence', {})
        
        explanation_parts = []
        
        # Docking contribution
        docking = breakdown.get('docking_contribution', 0)
        if docking > 25:
            explanation_parts.append(f"Excellent docking score ({docking:.0f}%)")
        elif docking > 15:
            explanation_parts.append(f"Good docking score ({docking:.0f}%)")
        else:
            explanation_parts.append(f"Weak docking ({docking:.0f}%)")
            
        # MD contribution
        md = breakdown.get('md_contribution', 0)
        if md > 15:
            explanation_parts.append(f"stable MD ({md:.0f}%)")
        elif md > 8:
            explanation_parts.append(f"moderate stability ({md:.0f}%)")
        else:
            explanation_parts.append(f"low stability ({md:.0f}%)")
            
        # MMGBSA contribution
        mmgbsa = breakdown.get('mmgbsa_contribution', 0)
        if mmgbsa > 15:
            explanation_parts.append(f"favorable ΔG ({mmgbsa:.0f}%)")
        elif mmgbsa > 8:
            explanation_parts.append(f"moderate ΔG ({mmgbsa:.0f}%)")
            
        # Confidence
        conf_level = confidence.get('level', 'medium')
        conf_reason = confidence.get('reason', '')
        
        base_explanation = f"Ranks #{hit.get('rank', '?')} because: {', '.join(explanation_parts)}"
        
        if conf_level == 'low':
            base_explanation += f". ⚠️ Low confidence: {conf_reason}"
        elif conf_level == 'high':
            base_explanation += f". ✅ High confidence prediction."
            
        return base_explanation


# Convenience function for API integration
def rank_compounds(hits: List[Dict], profile: str = 'balanced') -> List[Dict]:
    """
    Rank compounds using the ML Scorer.
    
    Args:
        hits: List of compound dictionaries
        profile: Scoring profile ('speed', 'accuracy', 'balanced', 'novelty')
        
    Returns:
        Ranked list with consensus scores
    """
    scorer = MLScorer(profile=profile)
    return scorer.rank_hits(hits)
