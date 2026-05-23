import os
import numpy as np
import pandas as pd
import sqlite3
import torch

from datetime import datetime

from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
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
# SEQUENCE
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
# EMBEDDING
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
# LOAD DATA
# =========================

def load_train():
    conn = sqlite3.connect("data/labels.sqlite")
    df = pd.read_sql_query("SELECT * FROM peptides", conn)
    conn.close()

    X, y = [], []

    for i, r in enumerate(df.itertuples()):
        if i % 50 == 0:
            print(i)

        X.append(embed(r.sequence))
        y.append([getattr(r, c) for c in CLASSES])

    return np.array(X), np.array(y), df

# =========================
# MODEL
# =========================

def train(X_train, y_train):
    models_a, models_b = [], []

    for i in range(len(CLASSES)):

        m1 = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        m2 = XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=7
        )

        m1.fit(X_train, y_train[:, i])
        m2.fit(X_train, y_train[:, i])

        models_a.append(m1)
        models_b.append(m2)

    return models_a, models_b

# =========================
# PREDICT
# =========================

def predict(models_a, models_b, X):
    pa = np.zeros((len(X), len(CLASSES)))
    pb = np.zeros((len(X), len(CLASSES)))

    for i in range(len(CLASSES)):
        pa[:, i] = models_a[i].predict_proba(X)[:, 1]
        pb[:, i] = models_b[i].predict_proba(X)[:, 1]

    return pa, pb

# =========================
# CV (OOF)
# =========================

def run_kfold_training(X, y):

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    oof_a = np.zeros((len(X), len(CLASSES)))
    oof_b = np.zeros((len(X), len(CLASSES)))

    print("\nRunning 3-fold CV...")

    for fold, (tr, va) in enumerate(kf.split(X)):

        print(f"Fold {fold+1}/3")

        X_train, X_val = X[tr], X[va]
        y_train = y[tr]

        m_a, m_b = train(X_train, y_train)

        pa, pb = predict(m_a, m_b, X_val)

        oof_a[va] = pa
        oof_b[va] = pb

    return oof_a, oof_b

# =========================
# THRESHOLD
# =========================

def tune_thresholds(probs, y):

    thresholds = []

    for i in range(len(CLASSES)):

        best_t, best_f1 = 0.5, 0

        for t in np.arange(0.1, 0.7, 0.02):

            pred = (probs[:, i] > t).astype(int)

            f1 = f1_score(y[:, i], pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds.append(best_t)

    return thresholds

# =========================
# MAIN
# =========================

def main():

    # ---------- DATA ----------
    X, y, df = load_train()

    # ---------- CV ----------
    oof_a, oof_b = run_kfold_training(X, y)

    # ---------- PCA (FIT ON FULL ONLY FOR FINAL STACK FEATURES) ----------
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)

    # ---------- META MODEL ----------
    meta_models = []

    meta_probs = np.zeros((len(X), len(CLASSES)))

    for i in range(len(CLASSES)):

        X_meta = np.column_stack([
            oof_a[:, i],
            oof_b[:, i],
            X_pca
        ])

        meta = LogisticRegression(max_iter=2000)
        meta.fit(X_meta, y[:, i])

        meta_models.append(meta)

        meta_probs[:, i] = meta.predict_proba(X_meta)[:, 1]

    # ---------- ROC ----------
    print("\nROC AUC:")

    for i, c in enumerate(CLASSES):
        auc = roc_auc_score(y[:, i], meta_probs[:, i])
        print(c, auc)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y[:, i], meta_probs[:, i])
        auc = roc_auc_score(y[:, i], meta_probs[:, i])

        plt.plot(fpr, tpr, label=f"{c} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (OOF CV)")
    plt.legend()
    plt.grid(True)

    plt.show()

    # ---------- THRESHOLDS ----------
    thresholds = tune_thresholds(meta_probs, y)

    oof_pred = (meta_probs > thresholds).astype(int)

    print("\nMacro F1:", f1_score(y, oof_pred, average="macro"))
    print("Micro F1:", f1_score(y, oof_pred, average="micro"))

    # ---------- FINAL TRAIN ----------
    final_a, final_b = train(X, y)

    # ---------- TEST ----------
    TEST_DIR = "test_pdbs"

    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".pdb")])

    X_test = []

    for f in test_files:
        seq = pdb_to_seq(os.path.join(TEST_DIR, f))
        X_test.append(embed(seq))

    X_test = np.array(X_test)

    test_a, test_b = predict(final_a, final_b, X_test)

    X_test_pca = pca.transform(X_test)

    test_meta = np.zeros((len(X_test), len(CLASSES)))

    for i in range(len(CLASSES)):
        X_m = np.column_stack([
            test_a[:, i],
            test_b[:, i],
            X_test_pca
        ])

        test_meta[:, i] = meta_models[i].predict_proba(X_m)[:, 1]

    test_pred = (test_meta > thresholds).astype(int)

    # ---------- SUBMISSION ----------
    submission = pd.DataFrame()
    submission["ID"] = [f.replace(".pdb", "") for f in test_files]

    for i, c in enumerate(CLASSES):
        submission[c] = test_pred[:, i]

    out_dir = f"submissions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    submission.to_csv(f"{out_dir}/submission.csv", index=False)

    print("Saved:", submission.shape)

# =========================

if __name__ == "__main__":
    main()