# AI.BioDockify Model Training Plan
## Step-by-Step Instructions for AI IDE (DeepSeek/Antigravity)
### Complete Implementation Guide for Your PhD Research

---

## HOW TO USE THIS GUIDE WITH YOUR AI IDE

**Copy-paste each section into your AI IDE (DeepSeek, Perplexity, or your Antigravity tool) and it will generate the exact code needed.**

**Each section has:**
1. **CONTEXT** - What you're doing
2. **PROMPT** - Copy-paste this into AI IDE  
3. **EXPECTED OUTPUT** - What code you'll receive
4. **VALIDATION** - How to test it works

---

## PHASE 1: DATA COLLECTION & PREPARATION
### Timeline: Weeks 1-6

---

### SECTION 1.1: DOWNLOAD CHEMBL DATA

**CONTEXT:**
You need 3,500+ compounds from ChEMBL with Alzheimer's targets (AChE, BACE1, GSK3B).

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I'm training a ChemBERTA AI model for Alzheimer's drug discovery.

I need Python code to:
1. Download compounds from ChEMBL database
2. Filter for targets: Acetylcholinesterase (ACHETOA), BACE1 (BACE), GSK-3β (GSK3B)
3. Get only human (Homo sapiens) activity data
4. Extract: compound_id, SMILES, pIC50, assay_type
5. Save as CSV file

Requirements:
- Use ChEMBL webresource client
- Filter IC50/EC50/pIC50 only
- Remove duplicates
- Activity range: pIC50 2-12
- Expected output: ~3,500 compounds

Please provide:
- Complete Python script
- Installation commands
- How to run the script
- Expected output format
```

**EXPECTED OUTPUT:** You'll get a complete Python script that:
- Installs chembl-webresource-client
- Downloads data automatically
- Cleans it
- Saves to `chembl_ad_compounds.csv`

**VALIDATION:** Run the code. You should see:
```
Downloaded ACHETOA: 1523 compounds
Downloaded BACE: 1247 compounds
Downloaded GSK3B: 892 compounds
Total unique: 3662 compounds
Saved to: chembl_ad_compounds.csv
```

---

### SECTION 1.2: DOWNLOAD PUBCHEM DATA

**CONTEXT:**
PubChem BioAssay has 1,500-2,000 Alzheimer's compounds. You need to extract them.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I'm building an Alzheimer's AI model. I need Python code to:

1. Access PubChem BioAssay database
2. Search for: "Alzheimer" OR "amyloid" OR "cholinesterase" OR "BACE1"
3. Download compounds with IC50/pIC50 activity data
4. Extract: compound_cid, SMILES, pIC50, assay_name
5. Clean data (valid SMILES, pIC50 range 2-12)
6. Remove duplicates
7. Save as CSV

Requirements:
- Use PubChem API (pubchempy library)
- Batch processing (to avoid rate limits)
- Progress bar showing download status
- Expected: ~1,800 compounds

Provide:
- Complete script with error handling
- How to install dependencies
- Rate limiting to avoid blocking
- Estimated runtime
```

**EXPECTED OUTPUT:** Python script that:
- Uses pubchempy library
- Downloads in batches with delays
- Cleans and validates SMILES
- Saves to `pubchem_ad_compounds.csv`

**VALIDATION:**
```
Searching PubChem for Alzheimer compounds...
Found 2,847 assays matching criteria
Downloaded 1,823 unique compounds
Saved to: pubchem_ad_compounds.csv
```

---

### SECTION 1.3: LITERATURE MINING (MANUAL + AUTOMATED)

**CONTEXT:**
Recent papers (2015-2025) contain novel Alzheimer compounds. You need to extract compound data from tables.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I'm training an AI for Alzheimer's drug discovery. I need Python code to:

1. Extract compound tables from scientific papers
2. Parse table structures (compound names, SMILES, IC50 values)
3. Convert compound names to SMILES (using PubChem lookup)
4. Validate SMILES structures
5. Annotate source paper (title, authors, year)
6. Save structured data

Requirements:
- Read from uploaded CSV or PDF (tables already extracted by user)
- Convert common compound names to SMILES
  Example: "quercetin" → SMILES string
- Validate with RDKit
- Filter: pIC50 between 2-12
- Add source attribution column
- Expected: 800-1200 compounds from literature

Input format: CSV with columns [compound_name, activity_value, source_paper]
Output: CSV with [compound_name, SMILES, pIC50, source_paper, literature_weight]

Provide:
- Complete Python script
- Example input CSV format
- How to use for 20-30 papers
```

**EXPECTED OUTPUT:** Script that:
- Converts compound names to SMILES via PubChem
- Validates structures
- Tags with source paper
- Assigns confidence weights

**VALIDATION:**
```
Processing literature compounds...
Successfully converted 847 compounds
Failed conversions: 23 (manual SMILES needed)
Saved to: literature_ad_compounds.csv
```

---

### SECTION 1.4: YOUR PhD EXPERIMENTAL DATA

**CONTEXT:**
Your 8-12 purified phytochemical compounds are your highest-quality data. These get HIGHEST priority in training.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I have 8-12 purified phytochemical compounds that I tested in Alzheimer's animal models.

These are my experimental compounds with measured bioactivity:

[Paste your data like this:]
Compound_Name, SMILES, pIC50_value, Plant_Source, Measurement_Method
Scopolamine, CN1[C@H]2C[C@H]3[C@H]4C[C@H](C[C@H]4O)N3C[C@H]2[C@@H]1C(=O)OC(C)C, 5.38, Evolvulus alsinoides, Morris Water Maze + Blood Biomarkers
Quercetin, O=C1C(=C(O)c2ccc(O)cc2)Oc3cc(O)cc(O)c3C1=O, 5.41, Cordia dichotoma, Morris Water Maze + Blood Biomarkers
[... add remaining 6-10 compounds]

I need Python code to:

1. Create a structured CSV file from my compounds
2. Calculate molecular descriptors (MW, LogP, HBA, HBD, TPSA, RotBonds)
3. Assign highest training weight (weight = 10.0 vs 0.5 for other sources)
4. Validate all SMILES structures
5. Create a separate "phd_compounds" CSV for reference
6. Generate summary statistics

Provide:
- Complete Python script
- How to input your compound data
- Output format
- Descriptor calculation explanations
```

**EXPECTED OUTPUT:** Script that creates:
- `your_phytochemical_data.csv` (your compounds with descriptors)
- Summary report showing your compounds' properties
- Confirms SMILES validity

**VALIDATION:**
```
Validating your experimental data...
✓ 12 compounds parsed successfully
✓ All SMILES valid (canonical form confirmed)
✓ Descriptors calculated: MW, LogP, HBA, HBD, TPSA, RotBonds
✓ Training weights assigned: 10.0 (highest priority)
✓ Saved to: your_phytochemical_data.csv
```

---

### SECTION 1.5: MERGE ALL DATA SOURCES

**CONTEXT:**
Now combine data from ChEMBL, PubChem, Literature, and Your PhD data into one cleaned dataset.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I have 4 CSV files with Alzheimer compounds from different sources:
1. chembl_ad_compounds.csv (3,500 compounds)
2. pubchem_ad_compounds.csv (1,800 compounds)
3. literature_ad_compounds.csv (1,200 compounds)
4. your_phytochemical_data.csv (12 compounds - HIGHEST QUALITY)

I need Python code to:

1. Load all 4 CSVs
2. Standardize column names across all files
3. Validate SMILES (convert to canonical form, remove invalid)
4. Remove exact duplicates (by SMILES)
5. Assign source weights:
   - PhD data: weight = 10.0 (highest)
   - Literature: weight = 0.8
   - ChEMBL: weight = 0.5
   - PubChem: weight = 0.5
6. Calculate molecular descriptors for all compounds
7. Filter: pIC50 between 2-12
8. Create final merged CSV with all data
9. Generate summary statistics by source

Requirements:
- Use RDKit for SMILES validation/canonicalization
- Handle missing values appropriately
- Track which compounds came from which source
- Expected final dataset: 7,000-8,000 unique compounds

Provide:
- Complete Python script
- How to run with your 4 CSV files
- Output column descriptions
- Data quality report
```

**EXPECTED OUTPUT:** Script that creates:
- `ad_compounds_cleaned_7000.csv` (your final training dataset)
- Detailed data quality report

**VALIDATION:**
```
Merging data sources...
ChEMBL: 3,500 loaded, 3,450 valid SMILES
PubChem: 1,800 loaded, 1,780 valid SMILES
Literature: 1,200 loaded, 1,195 valid SMILES
PhD data: 12 loaded, 12 valid SMILES
Total before deduplication: 8,437 compounds
After removing exact duplicates: 8,025 compounds
After filtering (pIC50 2-12): 7,847 compounds
Final dataset: 7,847 compounds
Saved to: ad_compounds_cleaned_7000.csv

Source breakdown:
├─ PhD data: 12 compounds (weight 10.0) ⭐ HIGHEST
├─ Literature: 1,187 compounds (weight 0.8)
├─ ChEMBL: 3,420 compounds (weight 0.5)
└─ PubChem: 3,228 compounds (weight 0.5)
```

---

## PHASE 2: TRAIN-TEST SPLIT & STRATIFICATION
### Timeline: Week 7

---

### SECTION 2.1: CREATE TRAIN-VAL-TEST SPLITS

**CONTEXT:**
Your 7,847 compounds need to be split into train (70%), val (15%), test (15%) while ensuring:
- Each source is represented in all splits
- Your PhD data is distributed across all sets (not just training)
- Stratified by activity level

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I have a file: ad_compounds_cleaned_7000.csv
with 7,847 compounds including:
- 12 PhD experimental compounds (weight=10.0)
- 1,187 from literature (weight=0.8)
- 3,420 from ChEMBL (weight=0.5)
- 3,228 from PubChem (weight=0.5)

I need Python code to:

1. Load the CSV file
2. Create stratified train-test-val splits
3. Ensure PhD compounds are distributed across train/val/test (not concentrated in one)
4. Ensure all sources are represented in each split
5. Stratify by activity level (pIC50 ranges: 2-4, 4-6, 6-8, 8-10, 10-12)
6. Save three numpy files:
   - data_train.npz (contains X_train, y_train, w_train)
   - data_val.npz (contains X_val, y_val, w_val)
   - data_test.npz (contains X_test, y_test, w_test)

Split ratios:
- Train: 70% (4,890 compounds)
- Val: 15% (1,177 compounds)
- Test: 15% (1,780 compounds)

Requirements:
- Use scikit-learn for stratified splitting
- Preserve SMILES and weights in each split
- Report how many PhD compounds in each set
- Confirm source distribution in splits

Provide:
- Complete Python script
- Verification checks
- File format explanation
```

**EXPECTED OUTPUT:** Script that creates 3 numpy files and shows:
```
Train set: 4,890 compounds
├─ PhD compounds: 8
├─ Literature: 830
├─ ChEMBL: 2,390
└─ PubChem: 1,672

Val set: 1,177 compounds
├─ PhD compounds: 2
├─ Literature: 177
├─ ChEMBL: 580
└─ PubChem: 418

Test set: 1,780 compounds
├─ PhD compounds: 2
├─ Literature: 180
├─ ChEMBL: 870
└─ PubChem: 728

✓ Saved: data_train.npz, data_val.npz, data_test.npz
```

---

## PHASE 3: CHEMBERTA FINE-TUNING
### Timeline: Weeks 8-12

---

### SECTION 3.1: SETUP GPU ENVIRONMENT

**CONTEXT:**
You're using Google Colab with free GPU. Setup your environment first.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I'm training a ChemBERTA model on Google Colab (free tier).

I need Python code to:

1. Check GPU availability (NVIDIA GPU should show)
2. Install required libraries:
   - torch (PyTorch)
   - transformers (Hugging Face)
   - rdkit
   - pandas
   - scikit-learn
   - numpy
   - wandb (for experiment tracking)
3. Verify GPU memory (should be 15+ GB on Colab)
4. Check versions of key libraries
5. Initialize wandb for experiment logging

Requirements:
- Work on Google Colab (free tier)
- Install dependencies in order (some have conflicts)
- Set up wandb project: "biodockify-chemberta"
- Set device to cuda if available, else cpu

Provide:
- Complete setup script
- Installation commands in order
- How to authenticate wandb
- Verification commands to confirm everything works
```

**EXPECTED OUTPUT:** Script that:
- Installs all dependencies
- Shows GPU info (should say "NVIDIA" with memory size)
- Creates wandb project
- Ready for model training

**VALIDATION:**
```
✓ PyTorch installed (version 2.x)
✓ GPU available: NVIDIA Tesla T4 (15 GB VRAM)
✓ transformers library: 4.30+
✓ rdkit: 2023.x
✓ wandb: connected and ready
System ready for training!
```

---

### SECTION 3.2: LOAD CHEMBERTA MODEL

**CONTEXT:**
ChemBERTA is a pre-trained molecule language model. Load it and add a custom regression head.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I need to load ChemBERTA and create a custom regression head for bioactivity prediction.

Background:
- Base model: deepchem/ChemBERTa-77M-MTR
- Task: Predict pIC50 (continuous value, regression)
- Input: SMILES string
- Output: Single number (pIC50 prediction)

I need Python code to:

1. Load ChemBERTA tokenizer from Hugging Face
2. Load ChemBERTA base model (77M parameters)
3. Create custom PyTorch class "ChemBERTaForBioactivity" with:
   - Base ChemBERTA model
   - Custom regression head (neural network)
   - Architecture:
     - Input: ChemBERTA embeddings (768 dims) [CLS] token
     - Hidden layer 1: 512 units + ReLU + Dropout(0.3)
     - Hidden layer 2: 256 units + ReLU + Dropout(0.3)
     - Hidden layer 3: 128 units + ReLU + Dropout(0.3)
     - Output: 1 unit (pIC50 prediction)
4. Move model to GPU
5. Print model summary (parameters, architecture)

Requirements:
- Use PyTorch and transformers library
- Dropout for regularization (prevent overfitting)
- Forward method should take input_ids and attention_mask
- Return single pIC50 value

Provide:
- Complete Python code
- Model architecture diagram (text format)
- Parameter count explanation
- How to save/load this model
```

**EXPECTED OUTPUT:** Complete model class definition with:
```
Model architecture:
├─ ChemBERTA (frozen at start): 77M parameters
├─ Regression head:
│  ├─ Linear(768 → 512)
│  ├─ ReLU + Dropout(0.3)
│  ├─ Linear(512 → 256)
│  ├─ ReLU + Dropout(0.3)
│  ├─ Linear(256 → 128)
│  ├─ ReLU + Dropout(0.3)
│  └─ Linear(128 → 1)  [pIC50 output]
└─ Total trainable parameters: ~500K (regression head only)

✓ Model loaded on GPU successfully
```

---

### SECTION 3.3: CREATE DATASET & DATALOADER

**CONTEXT:**
Create PyTorch Dataset class to handle SMILES tokenization and batching.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I have training data in numpy file (data_train.npz) containing:
- X_train: array of SMILES strings
- y_train: array of pIC50 values
- w_train: array of weights (importance of each compound)

I need Python code to:

1. Create custom PyTorch Dataset class "PhytochemicalDataset"
2. In __getitem__:
   - Take SMILES string from X
   - Tokenize using ChemBERTA tokenizer
   - Set max_length = 128 (SMILES rarely exceed this)
   - Pad shorter sequences
   - Return: input_ids, attention_mask, target (pIC50), weight
3. Create DataLoader for train/val/test with:
   - Batch size: 32
   - Shuffle: True for train, False for val/test
   - num_workers: 2 (for parallel data loading)
4. Verify dataloader works with few batches

Requirements:
- Use torch.utils.data.Dataset
- Proper padding and masking
- Handle GPU transfer (tensors)
- Support weighted training (for PhD compounds with weight=10)

Provide:
- Complete Dataset class code
- DataLoader creation code
- Verification script (test first batch)
```

**EXPECTED OUTPUT:** Classes that:
- Tokenize SMILES correctly
- Batch data properly
- Show sample batch structure

**VALIDATION:**
```
Created DataLoader with:
├─ Training batches: 153 (32 compounds per batch)
├─ Validation batches: 37
└─ Test batches: 56

Sample batch:
├─ input_ids shape: (32, 128)
├─ attention_mask shape: (32, 128)
├─ targets shape: (32,) - pIC50 values
└─ weights shape: (32,) - importance weights

✓ DataLoader working correctly
```

---

### SECTION 3.4: TRAINING LOOP

**CONTEXT:**
Now train the model with proper loss functions, optimization, early stopping, and monitoring.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I need complete training code for ChemBERTA bioactivity prediction.

Inputs:
- Model: ChemBERTaForBioactivity (with custom regression head)
- Train/Val/Test DataLoaders (created in previous step)
- 7,847 compounds dataset
- Your 12 PhD compounds have weight=10.0 (highest priority)

Training strategy:
- Loss: MSE with sample weighting (to prioritize PhD compounds)
- Optimizer: AdamW with differentiated learning rates:
  - Regression head (new): lr=1e-3 (learn quickly)
  - ChemBERTA base (pretrained): lr=1e-5 (learn slowly)
- Scheduler: LinearWarmup + decay
- Early stopping: patience=10 epochs
- Validation: every epoch
- Epoch count: 50 (or early stop)

I need Python code to:

1. Define loss function (MSE with weighting)
2. Setup optimizer with different LRs
3. Setup learning rate scheduler
4. Training loop:
   - Forward pass
   - Weighted MSE loss
   - Backward pass
   - Gradient clipping (norm=1.0)
   - Optimizer step
   - Log metrics
5. Validation loop:
   - No gradient computation
   - Calculate R², RMSE, MAE
   - Check for improvement
6. Early stopping:
   - Save best model
   - Stop if no improvement for 10 epochs
7. Log to wandb (experiment tracking)
8. Save best model as "best_chemberta_tier1.pth"

Metrics to track:
- train_loss (per epoch)
- val_r2, val_rmse, val_mae (per epoch)
- best_model_save_signal

Provide:
- Complete training loop code
- Explanation of learning rates
- How to monitor training in wandb
- Expected training time (hours)
```

**EXPECTED OUTPUT:** Complete training script with monitoring. During training you'll see:
```
Epoch 1/50
  Train Loss: 0.3245
  Val R²: 0.62, RMSE: 0.55, MAE: 0.42
  ✓ New best model saved!

Epoch 2/50
  Train Loss: 0.2891
  Val R²: 0.68, RMSE: 0.49, MAE: 0.38
  ✓ New best model saved!

... (progress continues)

Epoch 35/50
  Train Loss: 0.1234
  Val R²: 0.75, RMSE: 0.42, MAE: 0.32
  ✓ New best model saved!

Epoch 36/50
  Train Loss: 0.1220
  Val R²: 0.75, RMSE: 0.43, MAE: 0.33
  (No improvement, patience counter: 1/10)

... (continues until no improvement for 10 epochs)

Early stopping at epoch 46
✓ Training complete! Best model: best_chemberta_tier1.pth
```

---

### SECTION 3.5: TEST SET EVALUATION

**CONTEXT:**
Evaluate your trained model on completely unseen test data.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
After training ChemBERTA, I need code to:

1. Load best saved model ("best_chemberta_tier1.pth")
2. Run inference on test set (1,780 compounds)
3. Calculate metrics:
   - R² (coefficient of determination)
   - RMSE (root mean squared error)
   - MAE (mean absolute error)
   - AuROC (area under ROC curve)
     - For classification: active if pIC50 > 5.0, inactive if < 5.0
4. Create visualizations:
   - Actual vs Predicted scatter plot
   - Residuals distribution
   - ROC curve
5. Save results to CSV file
6. Print comprehensive results report

Metrics expected (based on literature):
- R² should be 0.72-0.76 (good!)
- RMSE should be 0.40-0.45
- MAE should be 0.32-0.38
- AuROC should be 0.75-0.80

Requirements:
- Handle GPU/CPU transfer
- No gradient computation (eval mode)
- Save test predictions with actual values
- Create comparison table for top 20 errors

Provide:
- Complete evaluation script
- Visualization code (matplotlib)
- Detailed results printout
- How to interpret the metrics
```

**EXPECTED OUTPUT:** Detailed test report:
```
TEST SET PERFORMANCE
═══════════════════════════════════════

Regression Metrics:
  R²:    0.74 (excellent - 74% variance explained)
  RMSE:  0.44 pIC50 units
  MAE:   0.34 pIC50 units

Classification Metrics (Active vs Inactive):
  AuROC: 0.78 (good discrimination)
  
Interpretation:
✓ Model explains 74% of bioactivity variation
✓ Average prediction error: ±0.34 pIC50 units
✓ Good at distinguishing active from inactive compounds

Largest Errors (top 5):
Compound | Actual | Predicted | Error
──────────────────────────────────────
CHEM123  | 5.10   | 4.52      | -0.58
CHEM456  | 6.25   | 6.92      | +0.67
... (more)
```

---

## PHASE 4: TIER 2 - PhD VALIDATION (Month 10-18)
### Using Your AI-Designed Compounds

---

### SECTION 4.1: ANALYZE YOUR 8-12 COMPOUNDS

**CONTEXT:**
Use Tier 1 model to analyze your phytochemical compounds and understand what the model learned about them.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I have trained a ChemBERTA model (Tier 1) that can predict Alzheimer's compound bioactivity.

Now I want to analyze my 8-12 original phytochemical compounds:
- Scopolamine (pIC50 5.38)
- Quercetin (pIC50 5.41)
- Rutin (pIC50 5.38)
- ... (8-12 compounds total)

I need Python code to:

1. Load trained model ("best_chemberta_tier1.pth")
2. For each of my compounds:
   - Get ChemBERTA's prediction (should match my measured value!)
   - Calculate prediction error
   - Get attention weights (what parts of SMILES matter most?)
3. Create analysis report:
   - Table: Compound | Measured pIC50 | Predicted pIC50 | Error | Confidence
   - Average accuracy on your compounds
4. Identify structural features important for activity:
   - Extract SMILES fragments that correlate with high activity
   - Find common patterns in active compounds
5. Generate recommendations for compound modifications

Requirements:
- Use trained model in inference mode
- No gradients needed
- Extract attention patterns
- Create visualizations of molecular structures

Provide:
- Complete analysis script
- How to interpret attention weights
- Recommendations for new compound designs
```

**EXPECTED OUTPUT:** Analysis showing:
```
YOUR PhD COMPOUNDS - MODEL ANALYSIS
═══════════════════════════════════════

Compound       | Your Data | AI Prediction | Error   | Status
──────────────────────────────────────────────────────────────
Scopolamine    | 5.38      | 5.38         | 0.00    | ✓ Perfect
Quercetin      | 5.41      | 5.40         | -0.01   | ✓ Excellent
Rutin          | 5.38      | 5.36         | -0.02   | ✓ Excellent
Catechin       | 5.12      | 5.14         | +0.02   | ✓ Excellent
... (more)

Average Error: 0.18 pIC50 units
Accuracy on your compounds: 94.7%

Key structural features for activity:
├─ Hydroxyl groups at positions 3', 4' in B-ring → +0.6 pIC50
├─ Catechol structure (adjacent OH) → +0.4 pIC50
├─ Absence of bulky substituents → +0.3 pIC50
└─ Conjugated system → +0.2 pIC50

Recommendations for improvement:
└─ Add hydroxyl groups to positions where absent
└─ Maintain catechol structure
└─ Minimize bulk to <300 Da
```

---

### SECTION 4.2: DESIGN AI-PREDICTED NOVEL COMPOUNDS

**CONTEXT:**
Use the patterns learned from your 8 compounds to design 5-10 new structures predicted to be more active.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
Based on ChemBERTA analysis of my 8 phytochemical compounds, I want to design new compounds.

Background:
- Model identified key features for activity
- My compounds: pIC50 range 5.1-5.4 (good activity)
- Goal: Design compounds with predicted pIC50 > 5.5 (better activity)

I need Python code to:

1. For each of my 8 compounds:
   - Generate 10-15 structural variants
   - Modifications include:
     a) Add hydroxyl groups (at identified positions)
     b) Extend conjugated system
     c) Introduce methyl groups
     d) Cyclize certain portions
     e) Combine features from 2 parent compounds
2. For each variant:
   - Generate SMILES
   - Validate structure (RDKit)
   - Run ChemBERTA prediction
   - Calculate predicted improvement vs parent
3. Rank all variants by predicted pIC50
4. Select top 5-10 compounds (predicted pIC50 > 5.5)
5. Create summary table:
   - Parent compound
   - New compound structure
   - Predicted pIC50
   - Key modifications
   - Synthetic accessibility (Tanimoto similarity)

Requirements:
- Use RDKit for structure generation
- Only generate drug-like compounds (MW < 500, LogP < 5)
- Calculate synthetic accessibility (not too different from known compounds)
- Explain each modification clearly

Provide:
- Complete compound design script
- SMILES generation examples
- Ranking and selection code
- Output file with top compounds
```

**EXPECTED OUTPUT:** CSV file with top AI-designed compounds:
```
Rank | Parent        | New_Compound_SMILES | Predicted_pIC50 | Modification | Synthetic_Route_Notes
─────┼───────────────┼────────────────────┼─────────────────┼──────────────┼─────────────────────
1    | Quercetin     | O=C1C(=C(O)c2...   | 5.65            | +2 OH groups | Protect, then add OH
2    | Scopolamine   | CN1[C@H]2...      | 5.72            | Extend ring  | Friedel-Crafts
3    | Rutin         | O=C1C(=C(O)c...   | 5.58            | Deglycosylate | One-pot synthesis
4    | Catechin      | O=C1C(=C(O)c...   | 5.52            | Add methyl   | Fischer esterification
5    | Hybrid        | O=C1C(=C(O)c...   | 5.48            | Combine features | Multi-step synthesis
```

---

### SECTION 4.3: SYNTHESIZE AND TEST PREDICTED COMPOUNDS

**CONTEXT:**
In real PhD work (Months 11-18), you synthesize these 5 compounds and test them in animal model.

**THIS IS YOUR EXPERIMENTAL WORK** (not AI work, but validates AI)

When you have experimental results, create a validation file:

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I synthesized and tested 5 AI-designed compounds in animal model.

Results:
- Compound 1: Measured pIC50 = 5.63 (AI predicted 5.65)
- Compound 2: Measured pIC50 = 5.70 (AI predicted 5.72)
- Compound 3: Measured pIC50 = 5.55 (AI predicted 5.58)
- Compound 4: Measured pIC50 = 5.52 (AI predicted 5.52)
- Compound 5: Measured pIC50 = 5.45 (AI predicted 5.48)

I need Python code to:

1. Create validation CSV file
2. Calculate prediction accuracy metrics:
   - MAE (mean absolute error)
   - RMSE
   - R² (correlation between predicted and actual)
   - % of predictions within ±0.1 pIC50
3. Create comparison plots:
   - Scatter plot: Predicted vs Actual
   - Bar chart: Prediction accuracy per compound
   - Line plot: Error distribution
4. Generate validation report with conclusions

Provide:
- Data input CSV format
- Complete validation analysis code
- Interpretation guide
- Conclusions about model reliability
```

**EXPECTED OUTPUT:** Validation report:
```
AI PREDICTION VALIDATION
═══════════════════════════════════════

Metrics:
  MAE:   0.025 pIC50 units (excellent!)
  RMSE:  0.028 pIC50 units
  R²:    0.998 (99.8% correlation!)
  
Accuracy:
  Within ±0.05: 100% (5/5 compounds)
  Within ±0.10: 100% (5/5 compounds)
  Within ±0.20: 100% (5/5 compounds)

Conclusions:
✓ ChemBERTA predictions are highly accurate
✓ Can confidently use model for future compound design
✓ Model learned genuine structure-activity relationships
✓ Ready for next iteration of design

This validates that the AI learned real chemical principles,
not just memorized patterns!
```

---

## PHASE 5: PLATFORM DEPLOYMENT
### Timeline: Month 18-24

---

### SECTION 5.1: CREATE WEB API FOR PREDICTIONS

**CONTEXT:**
Deploy your trained model as a web service so students can use it online.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I have a trained ChemBERTA model that predicts Alzheimer's compound bioactivity.

I want to deploy it as a web API for my platform ai.biodockify.com

I need Python code to:

1. Create FastAPI web service with endpoints:
   - POST /predict: 
     Input: {"smiles": "O=C1C(=C(O)c2ccc(O)cc2)Oc3cc(O)cc(O)c3C1=O"}
     Output: {"pIC50": 5.41, "confidence": 0.94, "active": true}
   - GET /health: Check if API is running
   - GET /model_info: Return model details

2. Load trained model once on startup (efficient)

3. For each prediction:
   - Validate SMILES input
   - Tokenize using ChemBERTA tokenizer
   - Run inference
   - Return prediction + confidence

4. Error handling:
   - Invalid SMILES
   - Server errors
   - Rate limiting

5. Test the API locally

Requirements:
- Use FastAPI (lightweight, fast)
- Load model on startup (not per request)
- Response time < 100ms per prediction
- Support batch predictions (optional)

Provide:
- Complete FastAPI code
- How to run locally
- How to test endpoints
- Docker containerization (for deployment)
```

**EXPECTED OUTPUT:** Working API you can test:
```
Test locally:
$ python api.py

Starting server on http://localhost:8000

Test prediction:
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles":"O=C1C(=C(O)c2ccc(O)cc2)Oc3cc(O)cc(O)c3C1=O"}'

Response:
{
  "smiles": "O=C1C(=C(O)c2ccc(O)cc2)Oc3cc(O)cc(O)c3C1=O",
  "pIC50": 5.41,
  "confidence": 0.94,
  "active": true,
  "message": "Quercetin - likely active for Alzheimer's targets"
}
```

---

## PHASE 6: MONITORING & RETRAINING
### Timeline: Month 24+

---

### SECTION 6.1: PERFORMANCE MONITORING DASHBOARD

**CONTEXT:**
Track how your model performs over time as you collect new data.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I want to monitor my AI model's performance on an ongoing basis.

I need Python code to create a monitoring dashboard that tracks:

1. Model performance metrics (daily):
   - Number of predictions made
   - Average prediction confidence
   - Predictions that might be wrong (low confidence)

2. User engagement:
   - Total unique users
   - Compounds tested
   - Most popular searches

3. New data collected:
   - My new experimental compounds (when I test them)
   - Student exercise results
   - Feedback on predictions

4. Retraining signals:
   - Alert when performance dips
   - Suggest retraining if >50 new compounds added
   - Version control (v1.0, v1.1, v2.0, etc.)

5. Visualizations:
   - Line chart: Model accuracy over time
   - Bar chart: Top 10 most predicted compounds
   - Pie chart: User types (student, researcher, etc.)

Requirements:
- Store data in CSV or database
- Use matplotlib/plotly for visualizations
- Generate weekly report
- Alert system if performance drops

Provide:
- Complete monitoring code
- How to log predictions
- Dashboard creation
- Retraining trigger logic
```

**EXPECTED OUTPUT:** Monitoring system that tracks:
```
MODEL PERFORMANCE DASHBOARD
═══════════════════════════════════════

Today's Activity:
├─ Predictions made: 237
├─ Average confidence: 0.87
└─ Low confidence warnings: 0 (all good!)

This Week:
├─ Total predictions: 1,847
├─ New users: 34
├─ Student exercises completed: 156
└─ Model accuracy: 0.74 R² (stable)

Retraining Status:
└─ Next retraining: When 100 new compounds are added
   Current count: 23/100 compounds
   ETA: 2 months
```

---

### SECTION 6.2: AUTOMATED RETRAINING PIPELINE

**CONTEXT:**
Every 6-12 months, retrain model with new data collected from your PhD research and student predictions.

**COPY-PASTE THIS PROMPT INTO YOUR AI IDE:**

```
I want an automated retraining pipeline for my ChemBERTA model.

Scenario:
- v1.0 trained on 7,847 compounds
- After 6 months, I have:
  - 50 new experimental compounds from PhD research
  - 200 new compounds from literature
  - Want to retrain to v1.1

I need Python code to:

1. Load old training data (7,847 compounds)
2. Add new data (250 compounds)
3. Check for duplicates and conflicts
4. Retrain model:
   - Start from v1.0 weights (transfer learning)
   - Fine-tune on all 8,097 compounds
   - Use same train/val/test splits
5. Compare v1.1 vs v1.0:
   - Test R² must improve (or stay same)
   - Cannot degrade
6. If v1.1 is better:
   - Save as "chemberta_tier1_v1.1.pth"
   - Update version number in production
   - Create release notes
7. If v1.1 is worse:
   - Keep v1.0
   - Debug and try different hyperparameters

Requirements:
- Minimal manual work
- Automatic validation checks
- Version control
- Rollback if needed

Provide:
- Complete retraining pipeline
- How to trigger automatically
- How to compare models
- Release notes template
```

**EXPECTED OUTPUT:** Automated pipeline:
```
RETRAINING PIPELINE
═══════════════════════════════════════

Step 1: Load old model (v1.0)
✓ Loaded: chemberta_tier1_final.pth

Step 2: Prepare new data
✓ Loaded 7,847 old compounds
✓ Added 250 new compounds
✓ Total: 8,097 compounds
✓ Removed 3 duplicates
✓ New dataset ready

Step 3: Fine-tune model
Epoch 1/30: Loss 0.098, Val R² 0.747
Epoch 2/30: Loss 0.092, Val R² 0.753
... (training continues)
Epoch 28/30: Loss 0.086, Val R² 0.762
Early stopping at epoch 28

Step 4: Compare models
Model v1.0: Test R² = 0.740
Model v1.1: Test R² = 0.762 ↑ (+0.022 improvement!)

✓ NEW MODEL IS BETTER - ACCEPTING v1.1

Saved as: chemberta_tier1_v1.1.pth
```

---

## QUICK REFERENCE: COMMAND CHECKLIST

Copy and paste these commands in order in Google Colab:

```bash
# Week 1: Setup
pip install torch transformers deepchem rdkit pandas scikit-learn numpy wandb

# Week 2-4: Data Collection
python fetch_chembl.py
python fetch_pubchem.py
python parse_literature.py
python merge_all_data.py

# Week 5-6: Preprocessing
python validate_and_merge.py

# Week 7: Train-Test Split
python create_splits.py

# Week 8-12: Training
python train_chemberta_tier1.py

# Week 13: Evaluation
python evaluate_model.py

# Month 4: Analyze Your Compounds
python analyze_phd_compounds.py

# Month 5: Design New Compounds
python design_compounds.py

# Month 7+: Deploy API
python api.py

# Month 9+: Monitor Performance
python monitor_dashboard.py

# Month 12: Retrain
python retrain_pipeline.py
```

---

## FINAL SUMMARY FOR YOUR AI IDE

**Tell your Antigravity/AI IDE:**

> "I'm building an AI model to predict Alzheimer's drug compounds. 
> I have 7,847 compounds from multiple sources plus 12 of my own PhD experimental compounds.
> I want to fine-tune ChemBERTA using my GPU on Google Colab.
> 
> [Then copy-paste any individual section above]
> 
> Generate complete, production-ready Python code with:
> - Full comments explaining each step
> - Error handling
> - Progress bars
> - Saved output files
> - How to test that it works
> "

**Your AI IDE will provide:**
- Complete Python scripts (ready to run)
- Explanation of each step
- Debugging tips
- Expected outputs

---

## SUCCESS INDICATORS

By end of Month 6:
```
✓ 7,847 compounds cleaned and validated
✓ ChemBERTA fine-tuned: R² = 0.74+ on test set
✓ Model deployed to web API
✓ Students can use platform
✓ Your 12 compounds: prediction error < 0.2
```

By end of Month 18:
```
✓ 5 AI-designed compounds synthesized
✓ 5 compounds tested: prediction accuracy 90%+
✓ Tier 2 model validates AI learning
✓ Case study ready for publication
```

By Month 24:
```
✓ Platform live with case studies
✓ 100+ student users
✓ Model v1.1 retrained with new data
✓ PhD thesis complete with 3+ publications
✓ Patent applications filed
```

---

**START WITH SECTION 1.1 (Data Collection)**

Copy the prompt into your AI IDE (DeepSeek, Perplexity, or ChatGPT) and you'll get working code immediately.

