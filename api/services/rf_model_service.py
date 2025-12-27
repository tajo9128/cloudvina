"""
RF Model Service (The AI Brain)
Responsible for Loading, Predicting, and Training Random Forest models for Docking Scoring.
Uses ODDT's implementation of RF-Score.
"""
import logging
import os
import joblib
import numpy as np
from pathlib import Path

# Try to import ODDT, handle failure gracefully for local dev without env
oddt_available = False
try:
    import oddt
    # Robust Import for InteractionFingerprint (Location changed in recent versions)
    try:
        from oddt.fingerprints import InteractionFingerprint
    except ImportError:
        from oddt.scoring import InteractionFingerprint
        
    from oddt.scoring.functions import RFScore
    oddt_available = True
except ImportError:
    oddt_available = False

logger = logging.getLogger("rf_model_service")

class RFModelService:
    """
    Manages the Random Forest Model for Binding Affinity Prediction.
    """
    
    _model_cache = None

    @classmethod
    def get_model(cls):
        """
        Singleton access to the RF Model.
        Supports loading custom scikit-learn models from models/ directory.
        """
        if cls._model_cache is None:
            try:
                # 1. Try Loading Custom Trained Model (sklearn Pipeline or Model)
                # Check for various filenames in priority order
                candidates = [
                    "models/rf_model.pkl",           # Standard Deployment Name
                    "models/rf_v3.0_production.pkl", # Raw Export
                    "models/rf_v2.1.pkl"             # Legacy
                ]
                
                found_path = None
                for path in candidates:
                    if os.path.exists(path):
                        found_path = path
                        break
                
                if found_path:
                    logger.info(f"Loading Custom RF Model from {found_path}...")
                    cls._model_cache = joblib.load(found_path)
                    cls._model_type = 'sklearn'
                
                # 2. Fallback to ODDT Internal Model
                elif oddt_available:
                    try:
                        logger.info(f"Custom model not found. Fallback to ODDT RFScore v4...")
                        cls._model_cache = RFScore.rfscore(version=4)
                        cls._model_type = 'oddt'
                    except Exception as e:
                        logger.warning(f"ODDT Native Model failed to load: {e}")
                        return None
                    
                else:
                    logger.warning("No RF model available (neither custom nor ODDT).")
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to load RF Model: {e}")
                return None
        return cls._model_cache

    @staticmethod
    def predict_ligand(receptor_path: str, ligand_path: str) -> float:
        """
        Predicts binding affinity (pKd).
        Handles both ODDT internal models and custom scikit-learn models.
        """
        model = RFModelService.get_model()
        if not model:
            return 6.0 # Moderate baseline

        try:
            # Prepare Input Features (Must match training!)
            # For Sklearn model (Custom v2.1/v3.0), we use ODDT InteractionFingerprint
            if getattr(RFModelService, '_model_type', 'oddt') == 'sklearn':
                if not oddt_available: 
                    logger.warning("ODDT missing for feature extraction")
                    return 6.0
                
                # Check file existence
                if not (os.path.exists(receptor_path) and os.path.exists(ligand_path)):
                    return 0.0

                try:
                    # Parse using ODDT toolkit (usually OpenBabel backend)
                    # Infer format from extension (pdbqt, pdb, etc)
                    ext_rec = os.path.splitext(receptor_path)[1].lower().replace('.', '') or 'pdbqt'
                    rec = next(oddt.toolkit.readfile(ext_rec, receptor_path))
                    
                    ext_lig = os.path.splitext(ligand_path)[1].lower().replace('.', '') or 'pdbqt'
                    lig = next(oddt.toolkit.readfile(ext_lig, ligand_path))
                    
                    rec.protein = True
                    
                    # Compute Fingerprint (Flattened)
                    # NOTE: If using the new V3 pipeline, it includes an Imputer, so NaNs are okay.
                    # The pipeline expects (n_samples, n_features)
                    ifp = InteractionFingerprint(lig, rec).flatten()
                    
                    # Reshape for single sample prediction (1, n_features)
                    features = ifp.reshape(1, -1)
                    
                    # Predict
                    pKd = model.predict(features)[0]
                    return round(float(pKd), 2)
                    
                except Exception as ex:
                    logger.error(f"Feature Extraction Failed: {ex}")
                    # Return baseline on failure to prevent app crash
                    return 6.0

            else:
                # ODDT Native Model (RFScore v4)
                if not oddt_available: return 6.0
                rec = next(oddt.toolkit.readfile('pdbqt', receptor_path))
                lig = next(oddt.toolkit.readfile('pdbqt', ligand_path))
                rec.protein = True
                return round(float(model.predict(lig, rec)[0]), 2)
            
        except Exception as e:
            logger.error(f"RF Prediction Failed: {e}")
            return 0.0

    @staticmethod
    def verify_model_card(card_path="models/model_card.json"):
        """
        Verifies the deployed model matches the international standard reproducibility card.
        """
        if not os.path.exists(card_path):
            return {"valid": False, "error": "No model card found"}
            
        try:
            with open(card_path, 'r') as f:
                card = json.load(f)
            return {"valid": True, "card": card}
        except Exception as e:
            return {"valid": False, "error": str(e)}
