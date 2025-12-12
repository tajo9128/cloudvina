"""
ChemBERTa QSAR Service for AI.BioDockify

This service provides:
1. Loading pre-trained ChemBERTa models from Hugging Face Hub (Legacy support)
2. SMILES-based bioactivity prediction (RandomForest + RDKit)
3. Support for multiple disease targets
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import os
from typing import List, Dict, Any, Union, Optional

# Heavy Libraries
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score

# Transformers (Optional/Legacy for specific disease models)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/torch not available. Deep Learning models disabled.")

class QSARService:
    """
    QSAR Prediction Service using Hybrid approach:
    1. RandomForest + Morgan Fingerprints (User Trained Models)
    2. ChemBERTa (Pre-trained Disease Models)
    """
    
    # Map of disease targets to their HuggingFace model repos
    DISEASE_MODELS = {
        "alzheimers": "tajo9128/biodockify-ai-alzheimers",
        "cancer": "tajo9128/biodockify-ai-cancer",
        "diabetes": "tajo9128/biodockify-ai-diabetes",
        "parkinson": "tajo9128/biodockify-ai-parkinson",
        "cardiovascular": "tajo9128/biodockify-ai-cardiovascular",
    }
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        self.device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.n_estimators = 100
        self.random_state = 42
        print(f"QSARService initialized. Device: {self.device}")
    
    # --- Part 1: Pre-trained Disease Models (ChemBERTa) ---

    def _load_model(self, disease_target: str):
        """Lazy-load a disease model from HF Hub."""
        if disease_target in self.loaded_models:
            return  # Already loaded
        
        if not TRANSFORMERS_AVAILABLE:
            print(f"Mock loading model for {disease_target}")
            return
            
        repo_id = self.DISEASE_MODELS.get(disease_target.lower())
        if not repo_id:
            # Silent fail for unknown targets, fall back to RF logic if needed
            return
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForSequenceClassification.from_pretrained(repo_id)
            model.to(self.device)
            model.eval()
            
            self.loaded_tokenizers[disease_target] = tokenizer
            self.loaded_models[disease_target] = model
        except Exception as e:
            print(f"Failed to load {disease_target} model: {e}")

    def predict_activity(
        self, 
        smiles_list: List[str], 
        disease_target: str = "alzheimers"
    ) -> List[Dict[str, Any]]:
        """
        Deep Learning Prediction for specific disease targets.
        """
        self._load_model(disease_target)
        results = []
        
        # Fallback if no DL model available
        if not TRANSFORMERS_AVAILABLE or self.loaded_models.get(disease_target) is None:
            # Use simple mock/heuristic if model missing (or could forward to RF)
            for smiles in smiles_list:
                results.append({
                    "smiles": smiles,
                    "score": 0.5,
                    "prediction": "Unknown",
                    "confidence": 0.0,
                    "disease_target": disease_target,
                    "mode": "fallback"
                })
            return results
        
        tokenizer = self.loaded_tokenizers[disease_target]
        model = self.loaded_models[disease_target]
        
        for smiles in smiles_list:
            try:
                inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    prob_active = probs[0][1].item()
                
                prediction = "Active" if prob_active > 0.5 else "Inactive"
                results.append({
                    "smiles": smiles,
                    "score": float(prob_active),
                    "prediction": prediction,
                    "confidence": float(max(prob_active, 1 - prob_active)),
                    "disease_target": disease_target,
                    "mode": "BioBERT"
                })
            except Exception:
                results.append({"smiles": smiles, "error": "Processing failed"})
                
        return results

    # --- Part 2: Custom User Training (RandomForest + RDKit) ---

    def to_fingerprints(self, smiles_list: List[str], radius: int = 2, nBits: int = 2048) -> np.ndarray:
        """Convert SMILES list to Morgan Fingerprints."""
        fps = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                    arr = np.zeros((1,))
                    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                    fps.append(arr)
                    valid_indices.append(i)
            except:
                continue
        
        if not fps:
            raise ValueError("No valid SMILES found in input.")

        return np.vstack(fps), valid_indices

    def train_model(self, smiles: List[str], targets: List[float], task_type: str = "regression") -> Dict[str, Any]:
        """Train a Random Forest model on provided data."""
        # 1. Featurization
        X, valid_idxs = self.to_fingerprints(smiles)
        y = np.array([targets[i] for i in valid_idxs])

        if len(y) < 5:
             raise ValueError("Not enough valid data points (min 5).")

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        # 3. Train
        metrics = {}
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, average='weighted'))
            }
        else:
            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
            }

        # 4. Retrain Full & Serialize
        model.fit(X, y)
        model_data = {
            "model": model, 
            "version": "1.0", 
            "type": "sklearn_rf",
            "task": task_type,
            "descriptor": "morgan_2048"
        }
        
        return {
            "metrics": metrics,
            "model_blob": pickle.dumps(model_data)
        }
    
    def predict(self, model_blob: bytes, smiles: List[str]) -> List[float]:
        """Load pickled model and predict on new SMILES."""
        model_data = pickle.loads(model_blob)
        model = model_data["model"]
        
        # Featurize (handle invalid smiles with 0-vectors to maintain index alignment)
        X = []
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    arr = np.zeros((1,))
                    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                    X.append(arr)
                else:
                    X.append(np.zeros((2048,))) 
            except:
                X.append(np.zeros((2048,))) 

        X = np.vstack(X)
        return model.predict(X).tolist()
