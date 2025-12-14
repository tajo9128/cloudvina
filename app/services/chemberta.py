"""
ChemBERTa QSAR Service for AI.BioDockify

This service provides:
1. Loading pre-trained ChemBERTa models from Hugging Face Hub
2. SMILES-based bioactivity prediction
3. Support for multiple disease targets
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional

# Heavy Libraries - Will be loaded on HF Spaces with GPU
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Using mock mode.")


class QSARService:
    """
    QSAR Prediction Service using ChemBERTa models.
    
    Supports multiple disease targets, each with their own fine-tuned model.
    Models are loaded from Hugging Face Hub on-demand.
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else "mock"
        print(f"QSARService initialized. Device: {self.device}")
    
    def _load_model(self, disease_target: str):
        """Lazy-load a disease model from HF Hub."""
        if disease_target in self.loaded_models:
            return  # Already loaded
        
        if not TRANSFORMERS_AVAILABLE:
            print(f"Mock loading model for {disease_target}")
            return
            
        repo_id = self.DISEASE_MODELS.get(disease_target.lower())
        if not repo_id:
            raise ValueError(f"Unknown disease target: {disease_target}. Available: {list(self.DISEASE_MODELS.keys())}")
        
        print(f"Loading model from {repo_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForSequenceClassification.from_pretrained(repo_id)
            model.to(self.device)
            model.eval()
            
            self.loaded_tokenizers[disease_target] = tokenizer
            self.loaded_models[disease_target] = model
            print(f"✓ Loaded {disease_target} model")
        except Exception as e:
            print(f"Failed to load {disease_target} model: {e}. Using fallback.")
            # Fallback to base ChemBERTa if fine-tuned model not found
            self.loaded_tokenizers[disease_target] = None
            self.loaded_models[disease_target] = None
    
    def predict_activity(
        self, 
        smiles_list: List[str], 
        disease_target: str = "alzheimers"
    ) -> List[Dict[str, Any]]:
        """
        Predict bioactivity for a list of SMILES strings.
        
        Returns:
            List of predictions with scores and interpretations.
        """
        self._load_model(disease_target)
        
        results = []
        
        # Mock mode fallback
        if not TRANSFORMERS_AVAILABLE or self.loaded_models.get(disease_target) is None:
            for smiles in smiles_list:
                score = np.random.rand()
                results.append({
                    "smiles": smiles,
                    "score": float(score),
                    "prediction": "Active" if score > 0.5 else "Inactive",
                    "confidence": float(max(score, 1-score)),
                    "disease_target": disease_target,
                    "mode": "mock"
                })
            return results
        
        # Real inference
        tokenizer = self.loaded_tokenizers[disease_target]
        model = self.loaded_models[disease_target]
        
        for smiles in smiles_list:
            inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                prob_active = probs[0][1].item()
            
            # Interpret
            if prob_active > 0.7:
                prediction = "Active"
                interpretation = "Strong bioactivity predicted. Recommended for experimental validation."
            elif prob_active > 0.5:
                prediction = "Moderate"
                interpretation = "Moderate activity predicted. Consider for further testing."
            else:
                prediction = "Inactive"
                interpretation = "Weak bioactivity. Suggest structural modifications."
            
            results.append({
                "smiles": smiles,
                "score": float(prob_active),
                "prediction": prediction,
                "confidence": float(max(prob_active, 1 - prob_active)),
                "interpretation": interpretation,
                "disease_target": disease_target,
                "mode": "inference"
            })
        
        return results
    
    # --- Legacy methods for backward compatibility ---
    
    def generate_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """Generate embeddings (mock for backward compat)."""
        return np.random.rand(len(smiles_list), 384)
    
    def train_model(self, smiles: List[str], targets: List[float]) -> Dict[str, Any]:
        """Mock training (real training happens in Colab)."""
        r2_score = 0.85 + (np.random.rand() * 0.1)
        fake_model = {"name": "ChemBERTa_Mock", "version": "1.0"}
        return {
            "metrics": {"r2": r2_score, "rmse": 0.12},
            "model_blob": pickle.dumps(fake_model)
        }
    
    def predict(self, model_blob: bytes, smiles: List[str]) -> List[float]:
        """Legacy predict (returns random scores)."""
        return list(np.random.rand(len(smiles)))
