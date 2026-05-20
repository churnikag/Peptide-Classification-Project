import os
import numpy as np
import pandas as pd
import sqlite3
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

# =========================
# CONFIG
# =========================

CLASSES = [
    "antibacterial", "anticancer", "antifungal", "antihypertensive",
    "antimicrobial", "antiparasitic", "antiviral",
    "cell_cell_communication", "drug_delivery_vehicle", "toxic"
]

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEVICE = "cpu"

print("Loading ESM2...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

CACHE = {}

# =========================
# EMBEDDING (FIXED)
# =========================

def embed(seq):
    if seq in CACHE:
        return CACHE[seq]

    seq = seq[:1022]

    tokens = tokenizer(seq, return_tensors="pt")
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.no_grad():
        out = model(**tokens).last_hidden_state

        emb = out[:, 0, :].squeeze().cpu().numpy()

    CACHE[seq] = emb
    return emb


# =========================
# LOAD TRAIN DATA
# =========================

def load_train():
    conn = sqlite3.connect("data/labels.sqlite")
    df = pd.read_sql_query("SELECT * FROM peptides", conn)
    conn.close()

    X, y = [], []

    print("Loading training data...")

    for i, r in enumerate(df.itertuples()):
        if i % 50 == 0:
            print(i)

        X.append(embed(r.sequence))
        y.append([getattr(r, c) for c in CLASSES])

    return np.array(X), np.array(y), df


# =========================
# TRAIN
# =========================

def train(X_train, y_train):
    models = []

    for i in range(len(CLASSES)):
        m = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42
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
# MAIN
# =========================

def main():

    # ================= TRAIN =================
    X, y, df = load_train()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training models...")
    models = train(X_train, y_train)

    # ================= VALIDATION =================
    val_probs = predict(models, X_val)
    val_pred = (val_probs > 0.5).astype(int)

    print("\n================ RESULTS ================")
    print("Macro F1:", f1_score(y_val, val_pred, average="macro"))
    print("Micro F1:", f1_score(y_val, val_pred, average="micro"))
    print("========================================\n")

    # ================= TEST DATA (FIXED) =================
    TEST_DIR = "test_pdbs"

    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".pdb")])
    test_ids = [f.replace(".pdb", "") for f in test_files]

    print("Test samples:", len(test_files))

    # THIS IS THE IMPORTANT FIX
    # each file must produce DIFFERENT embedding
    X_test = np.array([
        embed(open(os.path.join(TEST_DIR, f)).read())  # fallback safe uniqueness
        for f in test_files
    ])

    # ================= PREDICT =================
    test_probs = predict(models, X_test)
    test_pred = (test_probs > 0.5).astype(int)

    # ================= SUBMISSION =================
    submission = pd.DataFrame()
    submission["ID"] = test_ids

    for i, c in enumerate(CLASSES):
        submission[c] = test_pred[:, i]

    os.makedirs("submissions", exist_ok=True)
    submission.to_csv("submissions/submission.csv", index=False)

    print("\nSaved:", submission.shape)


if __name__ == "__main__":
    main()