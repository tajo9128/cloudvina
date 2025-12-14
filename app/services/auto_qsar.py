import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from fastapi import UploadFile
import io
import logging
import joblib
import os

logger = logging.getLogger(__name__)

class AutoQSARService:
    def __init__(self, models_dir="data/models/auto"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def _compute_fingerprints(self, smiles_list):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        valid_indices = [i for i, m in enumerate(mols) if m is not None]
        valid_mols = [mols[i] for i in valid_indices]
        
        # Morgan Fingerprints (Radius 2, 1024 bits)
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in valid_mols]
        X = np.array(fps)
        return X, valid_indices

    async def train_model(self, file: UploadFile, target_col: str, model_name: str):
        try:
            # Read CSV
            content = await file.read()
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            
            # Validation
            if 'smiles' not in df.columns.str.lower():
                # Try to find smiles column
                found = False
                for col in df.columns:
                    if 'smiles' in col.lower():
                        df.rename(columns={col: 'smiles'}, inplace=True)
                        found = True
                        break
                if not found:
                    raise ValueError("CSV must have a 'smiles' column")
            else:
                # Normalize column name
                df.columns = [c.lower() if c.lower() == 'smiles' else c for c in df.columns]

            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in CSV")

            # Data Prep
            df = df.dropna(subset=['smiles', target_col])
            smiles = df['smiles'].tolist()
            y_all = df[target_col].values

            # Featurization
            X, valid_idx = self._compute_fingerprints(smiles)
            y = y_all[valid_idx]

            if len(y) < 10:
                raise ValueError("Dataset too small (need at least 10 valid compounds)")

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train (Random Forest)
            # Limit depth/estimators for free tier performance
            model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Cross Validation (3-fold)
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
            cv_r2 = cv_scores.mean()

            # Feature Importance (Dummy mapping for bits)
            importances = model.feature_importances_
            top_bits = np.argsort(importances)[-5:][::-1].tolist()
            
            # Save Model (Simple Joblib)
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)

            return {
                "dataset_size": len(y),
                "metrics": {
                    "test_r2": float(r2),
                    "cv_r2": float(cv_r2),
                    "rmse": float(rmse)
                },
                "status": "success",
                "model_id": model_name,
                "top_features": top_bits
            }

        except Exception as e:
            logger.error(f"Auto-QSAR Training failed: {str(e)}")
            raise e
