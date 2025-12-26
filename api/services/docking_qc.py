"""
Docking Quality Control Validator

RULE 4: Random Forest "Truth Check"
Validates docking results against physical constraints before feeding to ML models.
Implements International-Level accuracy filters.
"""
import logging

logger = logging.getLogger("docking_qc")

class DockingQualityControl:
    """
    Post-docking validation filters to ensure "International Level" accuracy.
    Prevents garbage data from polluting Random Forest training sets.
    """
    
    @staticmethod
    def validate_result(result: dict) -> dict:
        """
        Validates a docking result against rigorous quality checks.
        
        Args:
            result: Dict with keys like 'vina_score', 'cnn_score', 'binding_affinity'
        
        Returns:
            Dict with added keys: 'qc_status', 'qc_warnings', 'qc_flags'
        """
        warnings = []
        flags = []
        status = "PASS"
        
        # Extract scores (handle various field names)
        vina_score = result.get('vina_score') or result.get('vina_affinity') or result.get('binding_affinity')
        cnn_score = result.get('cnn_score') or result.get('gnina_score')
        
        # FILTER 1: The "Clash" Detector
        # Positive Vina energy = severe steric repulsion
        if vina_score is not None:
            try:
                vina_val = float(vina_score)
                if vina_val > 0.0:
                    status = "REJECT"
                    flags.append("CLASH")
                    warnings.append(f"❌ Positive Vina Score ({vina_val:.2f} kcal/mol): Severe steric clash detected. Result is INVALID.")
                elif vina_val > -1.0:
                    flags.append("WEAK_BINDER")
                    warnings.append(f"⚠️ Near-Zero Affinity ({vina_val:.2f}): Likely surface interaction or decoy.")
            except (ValueError, TypeError):
                pass
        
        # FILTER 2: CNN Confidence Check (GNINA-specific)
        # CNNscore < 0.5 = model thinks it's a false positive
        if cnn_score is not None:
            try:
                cnn_val = float(cnn_score)
                if cnn_val < 0.5:
                    flags.append("CNN_LOW_CONFIDENCE")
                    warnings.append(f"⚠️ Low CNN Score ({cnn_val:.3f}): Deep learning model flags as potential false positive.")
                elif cnn_val > 0.8:
                    flags.append("CNN_HIGH_CONFIDENCE")
                    logger.info(f"✓ High CNN Confidence ({cnn_val:.3f})")
            except (ValueError, TypeError):
                pass
        
        # FILTER 3: Ligand Efficiency Check
        # LE = Affinity / Heavy Atom Count
        # Good binders: LE > 0.3 kcal/mol per heavy atom
        # Weak: LE < 0.2
        if vina_score is not None and result.get('heavy_atom_count'):
            try:
                vina_val = float(vina_score)
                heavy_atoms = int(result.get('heavy_atom_count', 20))  # Assume ~20 if missing
                
                # Vina scores are negative, so absolute value
                le = abs(vina_val) / heavy_atoms
                
                if le < 0.2:
                    flags.append("LOW_LIGAND_EFFICIENCY")
                    warnings.append(f"⚠️ Low Ligand Efficiency ({le:.3f}): Binding may not translate to real potency.")
                elif le > 0.4:
                    flags.append("HIGH_LIGAND_EFFICIENCY")
                    logger.info(f"✓ Excellent Ligand Efficiency ({le:.3f})")
                    
            except (ValueError, TypeError, ZeroDivisionError):
                pass
        
        # FILTER 4: Zero-Score Detection (Processing Failure)
        # If all scores are exactly 0.0, something went wrong
        all_zero = True
        for key in ['vina_score', 'cnn_score', 'binding_affinity', 'docking_score']:
            val = result.get(key)
            if val is not None and float(val) != 0.0:
                all_zero = False
                break
        
        if all_zero:
            status = "ERROR"
            flags.append("ZERO_SCORES")
            warnings.append("❌ All scores are zero: Likely processing failure or grid box error.")
        
        # Aggregate
        result['qc_status'] = status
        result['qc_warnings'] = warnings
        result['qc_flags'] = flags
        result['qc_passed'] = (status == "PASS")
        
        # Log summary
        if status == "REJECT":
            logger.warning(f"🚫 QC REJECT: Job {result.get('id', 'unknown')} - {', '.join(flags)}")
        elif warnings:
            logger.info(f"⚠️ QC PASS (with warnings): Job {result.get('id', 'unknown')} - {len(warnings)} warnings")
        else:
            logger.info(f"✓ QC PASS: Job {result.get('id', 'unknown')}")
        
        return result
    
    @staticmethod
    def should_include_in_training(result: dict) -> bool:
        """
        Determines if a result is clean enough for Random Forest training.
        
        Returns:
            True if result passes all critical filters, False otherwise.
        """
        if not result.get('qc_passed', False):
            return False
        
        # Additional strictness for ML training
        # Reject results with certain flags even if they "passed" validation
        reject_flags = {'CLASH', 'ZERO_SCORES'}
        flags_set = set(result.get('qc_flags', []))
        
        if reject_flags & flags_set:
            return False
        
        return True
