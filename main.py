import os
import sqlite3
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve, auc
from Bio.PDB import PDBParser

# =========================
# CONFIG
# =========================

CLASSES = [
    "antibacterial", "anticancer", "antifungal", "antihypertensive",
    "antimicrobial", "antiparasitic", "antiviral",
    "cell_cell_communication", "drug_delivery_vehicle", "toxic"
]

DB_PATH = "data/labels.sqlite"
TRAIN_PDB_DIR = "data/pdb"
TEST_PDB_DIR = "test_pdbs"

DEVICE = "cpu"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
esm_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
esm_model.eval()

CACHE = {}

# =========================
# EMBEDDING
# =========================

def embed(seq):
    seq = seq.strip().upper()
    if seq in CACHE:
        return CACHE[seq]

    seq = seq[:1022]
    tokens = tokenizer(seq, return_tensors="pt")
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.no_grad():
        out = esm_model(**tokens).last_hidden_state

        emb = torch.cat([
            out[:, 0, :],
            out.mean(dim=1),
            out.max(dim=1).values
        ], dim=1)

    emb = emb.squeeze().cpu().numpy().astype(np.float32)
    emb /= (np.linalg.norm(emb) + 1e-8)

    CACHE[seq] = emb
    return emb

# =========================
# STRUCTURE FEATURES
# =========================

def pdb_features(path):
    if not os.path.exists(path):
        return np.zeros(10, dtype=np.float32)

    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("x", path)
    except:
        return np.zeros(10, dtype=np.float32)

    coords = np.array(
        [a.get_coord() for a in structure.get_atoms() if a.get_name() == "CA"],
        dtype=np.float32
    )

    if len(coords) == 0:
        return np.zeros(10, dtype=np.float32)

    centroid = coords.mean(axis=0)
    spread = coords.std(axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)

    return np.concatenate([
        centroid,
        spread,
        [dists.max(), dists.min(), dists.mean(), len(coords)]
    ])

# =========================
# SEQUENCE FROM PDB
# =========================

def extract_sequence_from_pdb(path):
    aa3 = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
        "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
        "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
        "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }

    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("x", path)
    except:
        return ""

    seq = []
    seen = set()

    for chain in structure.get_chains():
        for res in chain.get_residues():
            rid = (chain.id, res.get_id())
            if rid in seen:
                continue
            seen.add(rid)

            aa = aa3.get(res.get_resname().strip())
            if aa:
                seq.append(aa)

    return "".join(seq)

# =========================
# LOAD TRAIN
# =========================

def load_train(mode):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM peptides", conn)
    conn.close()

    skip = set(CLASSES) | {"sequence"}
    id_col = next(c for c in df.columns if c not in skip and df[c].dtype == object)

    X, y = [], []

    for i, r in df.iterrows():
        seq_emb = embed(str(r["sequence"]))

        if mode == "seq_only":
            feat = seq_emb
        else:
            p = os.path.join(TRAIN_PDB_DIR, f"{r[id_col]}.pdb")
            feat = np.concatenate([seq_emb, pdb_features(p)])

        X.append(feat)
        y.append([int(r[c]) for c in CLASSES])

    return np.array(X), np.array(y)

# =========================
# LOAD TEST
# =========================

def load_test(mode):
    files = sorted([f for f in os.listdir(TEST_PDB_DIR) if f.endswith(".pdb")])

    ids, X = [], []

    for f in files:
        pid = f.replace(".pdb", "")
        seq = extract_sequence_from_pdb(os.path.join(TEST_PDB_DIR, f))

        if not seq:
            seq = "A" * 10

        emb = embed(seq)

        if mode == "seq_only":
            feat = emb
        else:
            feat = np.concatenate([emb, pdb_features(os.path.join(TEST_PDB_DIR, f))])

        ids.append(pid)
        X.append(feat)

    return ids, np.array(X)

# =========================
# MODEL
# =========================

def train_models(X, y):
    models = []
    for i in range(len(CLASSES)):
        m = LogisticRegression(max_iter=5000, class_weight="balanced")
        m.fit(X, y[:, i])
        models.append(m)
    return models


def predict(models, X):
    out = np.zeros((len(X), len(CLASSES)))
    for i, m in enumerate(models):
        out[:, i] = m.predict_proba(X)[:, 1]
    return out

# =========================
# CV
# =========================

def run_cv(X, y):
    kf = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    oof = np.zeros((len(X), len(CLASSES)))

    for tr, va in kf.split(X, y):
        m = train_models(X[tr], y[tr])
        oof[va] = predict(m, X[va])

    return oof

# =========================
# THRESHOLDS
# =========================

def tune_thresholds(probs, y):
    thr = []

    for i in range(len(CLASSES)):
        best_t, best_f = 0.5, 0

        for t in np.arange(0.1, 0.9, 0.01):
            pred = (probs[:, i] >= t).astype(int)
            f = f1_score(y[:, i], pred, zero_division=0)

            if f > best_f:
                best_f, best_t = f, t

        thr.append(best_t)

    return np.array(thr)

# =========================
# ROC
# =========================

def plot_roc(y, probs, name):
    plt.figure()

    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y[:, i], probs[:, i])
        score = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{c} AUC={score:.2f}")

    plt.plot([0,1],[0,1],'--')
    plt.title(name)
    plt.legend()
    plt.savefig(f"{name}_roc.png")
    plt.close()

# =========================
# PIPELINE
# =========================

def run_pipeline(mode):
    print("\nMODE:", mode)

    X, y = load_train(mode)

    # =========================
    # CV
    # =========================
    oof = run_cv(X, y)

    thr = tune_thresholds(oof, y)
    preds = (oof >= thr).astype(int)

    print("CV Macro F1:", f1_score(y, preds, average="macro"))

    plot_roc(y, oof, mode)

    # =========================
    # HOLDOUT (FIXED SCOPE)
    # =========================
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    hold_models = train_models(X_tr, y_tr)
    hold_probs = predict(hold_models, X_val)

    hold_pred = (hold_probs >= thr).astype(int)

    print("Holdout Macro F1:", f1_score(y_val, hold_pred, average="macro"))

    # =========================
    # FINAL TRAIN
    # =========================
    models = train_models(X, y)

    return models, thr

# =========================
# SUBMISSION
# =========================

def make_submission(mode, models, thr, file):
    ids, X = load_test(mode)
    probs = predict(models, X)

    rows = []
    for i, pid in enumerate(ids):
        rows.append([pid] + list((probs[i] >= thr).astype(int)))

    pd.DataFrame(rows, columns=["ID"] + CLASSES).to_csv(file, index=False)

# =========================
# MAIN
# =========================

def main():

    m1, t1 = run_pipeline("seq_only")
    make_submission("seq_only", m1, t1, "submission_seq_only.csv")

    m2, t2 = run_pipeline("seq_struct")
    make_submission("seq_struct", m2, t2, "submission_seq_struct.csv")

    print("DONE")

if __name__ == "__main__":
    main()