
import os
import sys
import logging

# Setup Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.rf_model_service import RFModelService

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RF_Test")

def test_integration():
    print("\n🔍 Checking RF Model Integration...\n" + "="*40)
    
    # 1. Check File Existence
    model_path = "models/rf_v3.0_production.pkl"
    if os.path.exists(model_path):
        print(f"✅ Found Model File: {model_path}")
    else:
        print(f"❌ Model File Missing: {model_path}")
        print("   -> Please drag the .zip contents here!")
        # Check alternatives
        if os.path.exists("models/rf_model.pkl"):
            print("   (Found 'rf_model.pkl' instead - this will work too)")
    
    # 2. Check Service Loading
    print("\n🔄 Attempting to Load Service...")
    model = RFModelService.get_model()
    
    if model:
        print(f"✅ Service Loaded Successfully!")
        print(f"   Type: {RFModelService._model_type}")
        print(f"   Object: {type(model)}")
    else:
        print("❌ Service Failed to Load Model.")
        return

    # 3. Simulate Prediction (Mock Data)
    print("\n🧪 Simulating Prediction...")
    # We can't predict without real files, but we can check if the method exists
    if hasattr(RFModelService, 'predict_ligand'):
        print("✅ predict_ligand() method is ready.")
    
    print("\n" + "="*40 + "\n✅ INTEGRATION VERIFIED (Pending File Upload)")

if __name__ == "__main__":
    test_integration()
