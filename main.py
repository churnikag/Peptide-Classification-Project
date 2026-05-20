import os
import numpy as np
import pandas as pd
import sqlite3
import torch

from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from Bio.PDB import PDBParser, PPBuilder

# =========================
# CONFIG
# =========================

CLASSES = [
    "antibacterial", "anticancer", "antifungal", "antihypertensive",
    "antimicrobial", "antiparasitic", "antiviral",
    "cell_cell_communication", "drug_delivery_vehicle", "toxic"
]

DEVICE = "cpu"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

print("Loading ESM2...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

CACHE = {}

# =========================
# PDB -> SEQUENCE
# =========================

def pdb_to_seq(path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", path)

    ppb = PPBuilder()
    seq = ""

    for pp in ppb.build_peptides(structure):
        seq += str(pp.get_sequence())

    return seq

# =========================
# ESM EMBEDDING
# CLS + MEAN POOLING
# =========================

def embed(seq):

    if seq in CACHE:
        return CACHE[seq]

    seq = seq[:1022]

    tokens = tokenizer(seq, return_tensors="pt")
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.no_grad():

        out = model(**tokens).last_hidden_state

        cls = out[:, 0, :]
        mean = out.mean(dim=1)
        maxpool = out.max(dim=1).values

        emb = torch.cat([cls, mean, maxpool], dim=1).squeeze().cpu().numpy()

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

    CACHE[seq] = emb

    return emb

# =========================
# LOAD TRAIN DATA
# =========================

def load_train():

    conn = sqlite3.connect("data/labels.sqlite")

    df = pd.read_sql_query("SELECT * FROM peptides", conn)

    conn.close()

    X = []
    y = []

    print("Loading training data...")

    for i, r in enumerate(df.itertuples()):

        if i % 50 == 0:
            print(i)

        X.append(embed(r.sequence))

        y.append([
            getattr(r, c) for c in CLASSES
        ])

    return np.array(X), np.array(y), df

# =========================
# TRAIN MODEL
# =========================

def train(X_train, y_train):

    models = []

    for i in range(len(CLASSES)):

        m = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        m.fit(X_train, y_train[:, i])

        models.append(m)

    return models

# =========================
# PREDICT
# =========================

def predict(models, X):

    probs = np.zeros((len(X), len(CLASSES)))

    for i, m in enumerate(models):

        probs[:, i] = m.predict_proba(X)[:, 1]

    return probs

# =========================
# 3-FOLD OOF TRAINING
# =========================

def run_kfold_training(X, y):

    kf = KFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    oof_probs = np.zeros((len(X), len(CLASSES)))

    print("\nRunning 3-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        print(f"Fold {fold + 1}/3")

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]

        models = train(X_train, y_train)

        val_probs = predict(models, X_val)

        oof_probs[val_idx] = val_probs

    return oof_probs

# =========================
# THRESHOLD TUNING
# =========================

def tune_thresholds(oof_probs, y):

    thresholds = []

    print("\nTuning thresholds...")

    for i in range(len(CLASSES)):

        best_t = 0.5
        best_f1 = 0

        for t in np.arange(0.1, 0.7, 0.02):

            preds = (oof_probs[:, i] > t).astype(int)

            score = f1_score(
                y[:, i],
                preds,
                zero_division=0
            )

            if score > best_f1:
                best_f1 = score
                best_t = t

        thresholds.append(best_t)

        print(
            f"{CLASSES[i]} -> "
            f"threshold={best_t:.2f}, "
            f"F1={best_f1:.4f}"
        )

    return thresholds

# =========================
# MAIN
# =========================

def main():

    # ================= TRAIN =================

    X, y, df = load_train()

    # ================= CV =================

    oof_probs = run_kfold_training(X, y)

    thresholds = tune_thresholds(oof_probs, y)

    # ================= EVALUATE =================

    oof_pred = np.zeros_like(oof_probs)

    for i in range(len(CLASSES)):

        oof_pred[:, i] = (
            oof_probs[:, i] > thresholds[i]
        ).astype(int)

    print("\n================ RESULTS ================")

    print(
        "Macro F1:",
        f1_score(y, oof_pred, average="macro")
    )

    print(
        "Micro F1:",
        f1_score(y, oof_pred, average="micro")
    )

    print("========================================\n")

    # ================= FINAL TRAIN =================

    print("Training final models...")

    final_models = train(X, y)

    # ================= TEST =================

    TEST_DIR = "test_pdbs"

    test_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.endswith(".pdb")
    ])

    test_ids = [
        f.replace(".pdb", "")
        for f in test_files
    ]

    print("Test samples:", len(test_files))

    X_test = []

    for i, f in enumerate(test_files):

        if i % 25 == 0:
            print(f"Embedding test {i}/{len(test_files)}")

        path = os.path.join(TEST_DIR, f)

        seq = pdb_to_seq(path)

        X_test.append(embed(seq))

    X_test = np.array(X_test)

    # ================= PREDICT =================

    test_probs = predict(final_models, X_test)

    test_pred = np.zeros_like(test_probs)

    for i in range(len(CLASSES)):

        test_pred[:, i] = (
            test_probs[:, i] > thresholds[i]
        ).astype(int)

    # ================= SUBMISSION =================

    submission = pd.DataFrame()

    submission["ID"] = test_ids

    for i, c in enumerate(CLASSES):

        submission[c] = test_pred[:, i]

    os.makedirs("submissions", exist_ok=True)

    submission.to_csv(
        "submissions/submission.csv",
        index=False
    )

    print("Saved:", submission.shape)

# =========================
# RUN
# =========================

if __name__ == "__main__":
    main()