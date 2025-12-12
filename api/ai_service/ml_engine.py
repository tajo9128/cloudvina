import os
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

class MLEngine:
    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.toxicity_model = None
        self.solubility_model = None
        
        # Initialize (Load or Cold Start)
        self.load_models()

    def _smiles_to_fp(self, smiles, n_bits=2048):
        """Convert SMILES to Morgan Fingerprint (ECFP4)"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            return np.array(fp)
        except:
            return None

    def load_models(self):
        """Load models from disk or trigger cold start training"""
        tox_path = os.path.join(self.model_dir, "toxicity_rf.joblib")
        sol_path = os.path.join(self.model_dir, "solubility_ridge.joblib")

        if os.path.exists(tox_path):
            self.toxicity_model = joblib.load(tox_path)
            print("loaded existing toxicity model.")
        else:
            print("No prediction model found. initializing COLD START training...")
            self._train_cold_start_models()

    def _train_cold_start_models(self):
        """
        Train lightweight models on tiny embedded dataset so the API works immediately.
        Data: A few known toxic vs non-toxic compounds.
        """
        # 1. Toxicity Data (0=Safe, 1=Toxic)
        tox_data = [
            ("CC(=O)Oc1ccccc1C(=O)O", 0), # Aspirin (Safe)
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 0), # Caffeine (Safe)
            ("O=C(O)C(N)C", 0), # Alanine (Safe)
            ("C1ccccc1", 1), # Benzene (Toxic)
            ("c1ccccc1[N+](=O)[O-]", 1), # Nitrobenzene (Toxic)
            ("C[Hg]Cl", 1) # Methylmercury (Toxic)
        ]
        
        X_tox = []
        y_tox = []
        for smi, label in tox_data:
            fp = self._smiles_to_fp(smi)
            if fp is not None:
                X_tox.append(fp)
                y_tox.append(label)

        # Train Random Forest
        self.toxicity_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        self.toxicity_model.fit(X_tox, y_tox)
        
        # Save it
        joblib.dump(self.toxicity_model, os.path.join(self.model_dir, "toxicity_rf.joblib"))
        print("Cold Start Toxicity Model trained and saved.")

    def predict_toxicity(self, smiles_list):
        """
        Predict toxicity for a list of SMILES.
        Returns: List of dicts {smiles, toxic_prob, is_toxic, confidence}
        """
        results = []
        
        # Bulk convert first
        valid_fps = []
        indices = []
        
        for idx, smi in enumerate(smiles_list):
            fp = self._smiles_to_fp(smi)
            if fp is not None:
                valid_fps.append(fp)
                indices.append(idx)
            else:
                results.append({
                    "smiles": smi, "error": "Invalid SMILES"
                })
        
        if not valid_fps:
            return results

        # Predict
        probs = self.toxicity_model.predict_proba(valid_fps)
        
        for i, original_idx in enumerate(indices):
            prob_toxic = probs[i][1]
            is_toxic = prob_toxic > 0.5
            
            # Simple confidence metric (dist from 0.5)
            confidence = abs(prob_toxic - 0.5) * 2 # 0.5->0, 1.0->1
            
            # Map back to results (needs proper ordering handling for production batching)
            # For simplicity, we just append here. Ideally match indices.
            result = {
                "smiles": smiles_list[original_idx],
                "probability_toxic": float(round(prob_toxic, 3)),
                "is_toxic": bool(is_toxic),
                "confidence": float(round(confidence, 2)),
                "model_version": "v1-coldstart"
            }
            results.append(result)
            
        return results

    def retrain_model(self, smiles, true_label):
        """
        Active Learning: Accept user feedback and retrain the model.
        true_label: 1 (Toxic) or 0 (Safe)
        """
        # 1. Convert new data point
        fp = self._smiles_to_fp(smiles)
        if fp is None:
            return {"status": "error", "message": "Invalid SMILES"}
            
        # 2. Add to cold start data (In a real DB, we'd query the full dataset)
        # For this Zero-Cost demo, we append to a temporary session list or re-generate base + new.
        # Simple approach: Re-load base data + new point.
        
        tox_data = [
            ("CC(=O)Oc1ccccc1C(=O)O", 0), 
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 0),
            ("O=C(O)C(N)C", 0),
            ("C1ccccc1", 1),
            ("c1ccccc1[N+](=O)[O-]", 1),
            ("C[Hg]Cl", 1)
        ]
        
        X_train = []
        y_train = []
        
        # Base Data
        for s, l in tox_data:
            X_train.append(self._smiles_to_fp(s))
            y_train.append(l)
            
        # New One (Weighted heavily? Or just added)
        X_train.append(fp)
        y_train.append(true_label)
        
        # 3. Retrain
        self.toxicity_model.fit(X_train, y_train)
        
        # 4. Save
        joblib.dump(self.toxicity_model, os.path.join(self.model_dir, "toxicity_rf.joblib"))
        
        return {
            "status": "success", 
            "message": f"Model retrained. New prediction for {smiles[:10]}... should be {'Toxic' if true_label else 'Safe'}"
        }
