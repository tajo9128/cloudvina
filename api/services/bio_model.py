import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

class BridgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.head = nn.Linear(768, 256) # Projection to 256 dim
    def forward(self, ids, mask):
        out = self.encoder(ids, mask).pooler_output
        return self.head(out)

class SpecialistModel(nn.Module):
    def __init__(self, bridge_model):
        super().__init__()
        self.bridge = bridge_model
        # Output LAYER (No Sigmoid here, we use BCEWithLogitsLoss logic internally, but for inference we sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.3),     
            nn.Linear(128, 1)
        )
    def forward(self, ids, mask):
        # We don't need gradients for inference
        with torch.no_grad(): 
            emb = self.bridge(ids, mask)
        return self.classifier(emb)

_tokenizer = None
_model = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return _tokenizer

def load_inference_model(path="biodockify_final.pth", device="cpu"):
    """
    Loads the full trained Tier 3 pipeline from the PTH file.
    """
    global _model
    if _model is not None:
        return _model

    try:
        print(f"Loading Tier 3 Model from {path}...")
        bridge = BridgeModel()
        model = SpecialistModel(bridge)
        
        # Load weights
        state_dict = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        _model = model
        print("✅ Tier 3 Model Loaded Successfully.")
        return _model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
