PEPTIDE / PROTEIN CLASSIFICATION PROJECT (ESM + STRUCTURE + MACHINE LEARNING)

---

OVERVIEW

This project builds a multi-label protein classification system that predicts biological functions of peptides using:

* Protein sequences extracted from PDB files
* ESM2 protein language model embeddings (Meta AI)
* Structural features from 3D protein coordinates
* Logistic Regression as the main classifier
* Multi-label stratified cross-validation
* Per-class threshold tuning
* Sequence-only and sequence+structure models

The model predicts multiple functional labels such as antibacterial, antiviral, anticancer, etc.

---

TOOLS USED (WHAT THEY DO)

GITHUB DESKTOP

* App used to connect local code to GitHub
* Tracks changes and uploads versions of your project

GITHUB

* Cloud platform for storing code
* Keeps full version history
* Allows rollback to older versions

PYCHARM

* Python code editor
* Used to write, run, and debug code
* Integrated with GitHub and databases

SQLITE PLUGIN (PYCHARM DATABASE TOOL)

* Lets you open and view SQLite databases inside PyCharm
* Used to inspect peptide label tables
* Helps run SQL queries like SELECT * FROM peptides

SQLITE DATABASE

* Stores structured data in table format
* Contains:

  * peptide IDs
  * amino acid sequences
  * class labels (antibacterial, antiviral, etc.)

PDB FILES (PROTEIN DATA BANK FILES)

* 3D structure files of proteins
* Contain atoms, coordinates, and amino acid information
* Used to extract structural and sequence features

ESM2 (EVOLUTIONARY SCALE MODEL)

* Deep learning model from Meta AI trained on millions of proteins
* Converts amino acid sequences into numeric embeddings
* Captures biological meaning of protein sequences

TRANSFORMERS LIBRARY (HUGGING FACE)

* Loads ESM model and tokenizer
* Converts sequences into embeddings

BIOPYTHON (Bio.PDB)

* Reads PDB structure files
* Extracts atoms, residues, and protein chains

SCIKIT-LEARN

* Machine learning library used for:

  * Logistic Regression
  * Cross-validation
  * F1 score and ROC metrics

ITERSTRAT (MULTILABEL STRATIFIED K-FOLD)

* Splits dataset into balanced folds for multi-label classification
* Ensures fair evaluation across all classes

MATPLOTLIB

* Used to plot ROC curves and visual results

NUMPY / PANDAS

* NumPy: numerical arrays and vectors
* Pandas: data tables and dataset handling

PYTORCH (TORCH)

* Runs deep learning models (ESM embeddings)
* Handles tensor computation

---

DATA STRUCTURE EXPLANATION

PDB FILES

Each row represents an atom:

* Atom Name (CA, CB, N, O): position inside amino acid
* Residue Name (THR, LYS, VAL): amino acid type
* Residue Number: position in protein chain
* XYZ Coordinates: 3D position in space
* Element Type (C, N, O): chemical element

KEY IDEA:
Atoms → Amino acids → Protein structure

---

SEQUENCE EXTRACTION FROM PDB

Example:
THR SER VAL ILE LYS

Step 1:
Read amino acids in chain order

Step 2:
Convert 3-letter codes to 1-letter:
THR → T
SER → S
VAL → V
ILE → I
LYS → K

Step 3:
Final sequence:
TSVIK...

KEY IDEA:
Each amino acid becomes one letter in a protein sequence

---

PROJECT PIPELINE

1. Load dataset from SQLite database
2. Extract protein sequences from PDB files
3. Generate ESM2 embeddings for sequences
4. Optionally extract structural features from PDB
5. Train Logistic Regression models (one per class)
6. Run 3-fold stratified cross-validation
7. Tune classification thresholds per label
8. Evaluate using F1 score and ROC-AUC
9. Train final model on full dataset
10. Generate predictions for test set

---

MY PROJECT PROGRESSION (COMMITS EXPLAINED)

1. SETUP (GITHUB + ENVIRONMENT)

* Connected PyCharm to GitHub for version control
* Installed SQLite plugin for database inspection
* Verified repository setup and workflow

2. DATA LOADING

* Added test PDB dataset
* Validated file structure and dataset integrity

3. BASELINE MODEL

* Built first end-to-end ML pipeline
* Implemented Logistic Regression baseline
* Created first working training → prediction system

4. ESM FEATURE ENGINEERING + IMBALANCE FIX

* Added ESM2 embeddings
* Improved biological representation of sequences
* Applied class weighting for imbalance handling

5. SEQUENCE + STRUCTURE FEATURES

* Extracted amino acid sequences from PDB
* Added structural features (centroid, spread, distances)
* Introduced 3-fold cross-validation
* Added per-class threshold tuning

6. MODEL EXPANSION

* Tested Logistic Regression and Random Forest
* Improved embedding strategy (CLS + mean + max pooling)
* Standardized feature pipeline

7. EVALUATION SYSTEM

* Added ROC curve plotting
* Computed AUC per class
* Analyzed model performance differences

8. MODEL EXPERIMENTATION PHASE

* Tested Logistic Regression, Random Forest, XGBoost
* Compared model and ensemble performance

9. VALIDATION IMPROVEMENTS

* Added train/validation holdout split
* Improved robustness of evaluation

10. FINAL ENSEMBLE SYSTEM

* Combined multiple model predictions
* Applied weighted ensemble approach
* Improved final prediction stability

11. FINAL CLEANUP

* Refined ESM embedding usage
* Confirmed per-class threshold tuning
* Maintained multilabel stratified cross-validation
* Final bug fixes before submission

---

FEATURE EXTRACTION

SEQUENCE FEATURES:

* ESM2 embeddings:

  * CLS token
  * Mean pooling
  * Max pooling

STRUCTURAL FEATURES:

* Centroid of atom coordinates
* Standard deviation (spread)
* Distance statistics (max, min, mean, count)

FINAL FEATURE VECTOR:
ESM embeddings + optional structural features

---

MODEL ARCHITECTURE

* Logistic Regression (baseline multi-label model)
* One model per class
* Class-weighted training
* Probability outputs per class
* Threshold tuned per class

---

TRAINING PROCESS

1. Load dataset from SQLite
2. Extract PDB sequences
3. Generate embeddings
4. Train models per class
5. Run 3-fold cross-validation
6. Tune thresholds
7. Train final model
8. Generate predictions

---

HOW TO RUN

Run:
python main.py

Outputs:

* submission_seq_only.csv
* submission_seq_struct.csv

---

FIGURES

Approach diagram:
[https://www.canva.com/design/DAHKfWs-fLo/m0chjHLL8t72TtjR3uBi4g/edit](https://www.canva.com/design/DAHKfWs-fLo/m0chjHLL8t72TtjR3uBi4g/edit)

Results figure:
[https://www.canva.com/design/DAHKfWs-fLo/m0chjHLL8t72TtjR3uBi4g/edit](https://www.canva.com/design/DAHKfWs-fLo/m0chjHLL8t72TtjR3uBi4g/edit)

---

LIMITATIONS

* ESM embeddings are computationally expensive
* PDB files may be incomplete or noisy
* Logistic Regression limits complex nonlinear learning
* Class imbalance affects rare labels
* Threshold tuning depends on validation quality

---

ASSUMPTIONS

* PDB files are correctly formatted
* Labels are accurate
* Training and test distributions are similar

---

TOOLS AND DEVELOPMENT SUPPORT

Parts of this project were developed with assistance from ChatGPT for:

* Debugging code issues
* Improving pipeline structure
* Writing documentation
* Rapid experimentation

Final code was tested and integrated into a working pipeline.

---

SUMMARY

PDB → sequence extraction → ESM embeddings → structural features → Logistic Regression → cross-validation → threshold tuning → final predictions

---