# Phase 4: Automation & Active Learning 🔄

**Goal**: Implement a **Self-Improving AI System** that learns from user feedback (Active Learning).
**Value**: This transforms the tool from a static predictor into an evolving asset that adapts to the user's specific chemical space.

## 1. Zero-Cost Active Learning Strategy
Since we cannot afford heavy GPU training loops, we will use **Online Learning** or **Incremental Retraining** with lightweight models (Random Forest/SGD).

### Workflow:
1.  **Inference**: User uploads molecules -> System predicts Toxicity.
2.  **Feedback**: User flags errors (e.g., "Model said Safe, but lab results say Toxic").
3.  **Retraining**: System updates the Random Forest weights using the new data point.
4.  **Version Control**: System saves `toxicity_v2.joblib`.

## 2. Implementation Steps

### Step A: Backend Logic (`api/ai_service/ml_engine.py`)
Add `retrain_model(new_data)` method:
1.  Load existing training dataset (or buffer).
2.  Append new user feedback.
3.  `model.fit()` (Fast on CPU for small/medium datasets).
4.  Save model.

### Step B: API Endpoints (`api/ai_service/main.py`)
- `POST /feedback`: Receives `{smiles, true_label}`.
- `POST /retrain`: Triggers the training process.
- `GET /model/history`: Returns accuracy history (v1: 89%, v2: 91%...).

### Step C: Frontend Studio (`web_ai`)
Replace Phase 4 placeholder with **"Active Learning Studio"**:
- **Review Queue**: Cards showing recent predictions.
- **Actions**: "Validate" (Green Check) or "Correct" (Red X -> flip label).
- **Progress Bar**: "5 new labels - Ready to Retrain".
- **Trigger**: "Improve AI" button.

## 3. Deliverable
A closed-loop system where the user interacts with the AI, and the AI gets smarter. 
**Real Enterprise capability on a $0 budget.**
