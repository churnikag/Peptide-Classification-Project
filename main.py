import sqlite3, pandas as pd
conn = sqlite3.connect("data/labels.sqlite")
df = pd.read_sql_query("SELECT * FROM peptides", conn)
conn.close()

classes = ["antibacterial","anticancer","antifungal","antihypertensive",
           "antimicrobial","antiparasitic","antiviral",
           "cell_cell_communication","drug_delivery_vehicle","toxic"]

for c in classes:
    pct = df[c].mean() * 100
    print(f"{c:30s}: {pct:.1f}% positive")

import os
import pickle
import numpy as np
import pandas as pd
import sqlite3
import torch

from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from Bio.PDB import PDBParser

# =========================
# CONFIG
# =========================

CLASSES = [
    "antibacterial", "anticancer", "antifungal", "antihypertensive",
    "antimicrobial", "antiparasitic", "antiviral",
    "cell_cell_communication", "drug_delivery_vehicle", "toxic"
]

# Paths matching the dataset description exactly
DB_PATH       = "data/labels.sqlite"
TRAIN_PDB_DIR = "data/pdb"
TEST_PDB_DIR  = "test_pdbs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
esm_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
esm_model.eval()

EMBED_CACHE = {}

# =========================
# SEQUENCE EMBEDDING
# =========================

def embed(seq):
    """Embed a peptide sequence using ESM2. Returns a normalized vector."""
    seq = seq.strip().upper()

    if seq in EMBED_CACHE:
        return EMBED_CACHE[seq]

    seq_trunc = seq[:1022]  # ESM2 max is 1024 tokens including special tokens

    tokens = tokenizer(seq_trunc, return_tensors="pt")
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.no_grad():
        out = esm_model(**tokens).last_hidden_state
        emb = torch.cat([
            out[:, 0, :],           # CLS token
            out.mean(dim=1),        # mean pool
            out.max(dim=1).values   # max pool
        ], dim=1)

    emb = emb.squeeze().cpu().numpy().astype(np.float32)

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    EMBED_CACHE[seq] = emb
    return emb


# =========================
# STRUCTURE FEATURES
# =========================

def pdb_features(pdb_path):
    """
    Extract geometric features from a PDB file using Cα atoms.
    Returns a 10-dim vector; zeros if file is missing or has no Cα atoms.
    """
    if not os.path.exists(pdb_path):
        return np.zeros(10, dtype=np.float32)

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("x", pdb_path)
    except Exception:
        return np.zeros(10, dtype=np.float32)

    coords = np.array(
        [a.get_coord() for a in structure.get_atoms() if a.get_name() == "CA"],
        dtype=np.float32
    )

    if len(coords) == 0:
        return np.zeros(10, dtype=np.float32)

    centroid  = coords.mean(axis=0)           # (3,)
    spread    = coords.std(axis=0)            # (3,)
    dists     = np.linalg.norm(coords - centroid, axis=1)
    max_dist  = dists.max()
    min_dist  = dists.min()
    mean_dist = dists.mean()
    n_res     = float(len(coords))

    return np.concatenate([centroid, spread, [max_dist, min_dist, mean_dist, n_res]])


# =========================
# SEQUENCE FROM PDB
# =========================

def extract_sequence_from_pdb(pdb_path):
    """
    Extract amino acid sequence from a PDB file by reading residue names in chain order.
    Returns a single-letter string, or empty string on failure.
    """
    aa3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("x", pdb_path)
    except Exception:
        return ""

    seq, seen = [], set()
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            res_id = (chain.id, residue.get_id())
            if res_id in seen:
                continue
            seen.add(res_id)
            aa = aa3to1.get(residue.get_resname().strip())
            if aa:
                seq.append(aa)

    return "".join(seq)


# =========================
# LOAD TRAINING DATA
# =========================

def load_train_data(mode):
    """
    Load all training peptides from data/train/labels.sqlite.
    Returns X (features), y (binary labels), DataFrame.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM peptides", conn)
    conn.close()

    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")

    # Detect the ID column: string dtype, not a class or known metadata column
    skip = set(CLASSES) | {"sequence", "length", "number_of_classes"}
    id_col = next(
        (c for c in df.columns if c not in skip and df[c].dtype == object),
        None
    )
    if id_col is None:
        raise ValueError(f"Cannot detect ID column. Columns: {list(df.columns)}")
    print(f"  ID column: '{id_col}'")

    X, y = [], []

    for i, r in df.iterrows():
        if i % 100 == 0:
            print(f"  Processing {i}/{len(df)} ...")

        seq_emb = embed(str(r["sequence"]))

        if mode == "seq_only":
            feat = seq_emb

        elif mode == "seq_struct":
            pdb_path = os.path.join(TRAIN_PDB_DIR, f"{r[id_col]}.pdb")
            feat = np.concatenate([seq_emb, pdb_features(pdb_path)])

        else:
            raise ValueError(f"Unknown mode: {mode}")

        X.append(feat)
        y.append([int(r[c]) for c in CLASSES])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), df


# =========================
# LOAD TEST DATA
# =========================

def load_test_data(mode):
    """
    Load test peptides from data/test/pdb/.
    Sequences are extracted from the PDB files directly.
    Returns ids (list) and X (feature matrix).
    """
    pdb_files = sorted(f for f in os.listdir(TEST_PDB_DIR) if f.endswith(".pdb"))

    if len(pdb_files) == 0:
        raise FileNotFoundError(f"No .pdb files found in '{TEST_PDB_DIR}'")

    print(f"  Found {len(pdb_files)} test PDB files")

    ids, X = [], []

    for i, fname in enumerate(pdb_files):
        if i % 100 == 0:
            print(f"  Processing test {i}/{len(pdb_files)} ...")

        pid      = fname.replace(".pdb", "")
        pdb_path = os.path.join(TEST_PDB_DIR, fname)

        seq = extract_sequence_from_pdb(pdb_path)
        if not seq:
            print(f"  WARNING: empty sequence for {pid}, using poly-A fallback")
            seq = "A" * 10

        seq_emb = embed(seq)

        if mode == "seq_only":
            feat = seq_emb
        elif mode == "seq_struct":
            feat = np.concatenate([seq_emb, pdb_features(pdb_path)])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ids.append(pid)
        X.append(feat)

    return ids, np.array(X, dtype=np.float32)


# =========================
# XGB MODELS
# =========================

def train_xgb(X, y):
    """Train one XGBClassifier per class with imbalance correction."""
    models = []
    for i, cls in enumerate(CLASSES):
        pos = y[:, i].sum()
        neg = len(y) - pos
        scale = neg / pos if pos > 0 else 1.0

        m = XGBClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X, y[:, i])
        models.append(m)
    return models


def predict_xgb(models, X):
    """Return probability matrix (n_samples, n_classes)."""
    probs = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    for i, m in enumerate(models):
        probs[:, i] = m.predict_proba(X)[:, 1]
    return probs


# =========================
# CROSS-VALIDATION
# =========================

def run_kfold(X, y, n_splits=3):
    """Multilabel stratified CV producing OOF probabilities."""
    kf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        print(f"  Fold {fold + 1}/{n_splits} ...")

        models = train_xgb(X[tr_idx], y[tr_idx])

        oof[va_idx] = predict_xgb(models, X[va_idx])

    return oof

# =========================
# THRESHOLD TUNING
# =========================

def tune_thresholds(probs, y):
    thresholds = []
    for i, cls in enumerate(CLASSES):
        pos_rate = y[:, i].mean()
        # Start threshold at the class positive rate — prevents always predicting 1
        best_t, best_f1 = max(0.3, pos_rate), 0.0
        for t in np.arange(max(0.1, pos_rate * 0.5), 0.95, 0.01):
            pred = (probs[:, i] >= t).astype(int)
            f    = f1_score(y[:, i], pred, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(best_t)
        print(f"    {cls:30s}: pos_rate={pos_rate:.2f}  threshold={best_t:.2f}  F1={best_f1:.4f}")
    return np.array(thresholds)

# =========================
# METRICS
# =========================

def print_metrics(y_true, y_pred, label=""):
    print(f"\n--- Metrics {label} ---")
    print(f"  Macro F1   : {f1_score(y_true, y_pred, average='macro',  zero_division=0):.4f}")
    print(f"  Micro F1   : {f1_score(y_true, y_pred, average='micro',  zero_division=0):.4f}")
    print(f"  Exact match: {np.mean(np.all(y_pred == y_true, axis=1)):.4f}")
    print("  Per-class F1:")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:30s}: {f1_score(y_true[:, i], y_pred[:, i], zero_division=0):.4f}")


# =========================
# PIPELINE
# =========================

def run_pipeline(mode, n_splits=3):
    """
    Full train + eval pipeline for one mode.
    Returns xgb_models, meta_models, pca, thresholds (tuned on OOF).
    """
    print(f"\n{'='*50}")
    print(f"PIPELINE MODE: {mode}")
    print(f"{'='*50}")

    # 1. Load data
    print("\n[1/5] Loading training data ...")
    X, y, df = load_train_data(mode)
    print(f"  X shape: {X.shape},  y shape: {y.shape}")

    # 2. OOF cross-validation
    print(f"\n[2/5] {n_splits}-fold cross-validation ...")
    oof_probs = run_kfold(X, y, n_splits=n_splits)

    # 3. Tune thresholds on OOF
    print("\n[3/5] Tuning thresholds on OOF probabilities ...")
    thresholds = tune_thresholds(oof_probs, y)
    print_metrics(y, (oof_probs >= thresholds).astype(int), label="[OOF XGB]")

    # 4. Stacking meta-model: LogReg on [oof_prob, PCA(X)]
    print("\n[4/5] Fitting PCA + stacking meta-model ...")
    n_components = min(50, X.shape[1], X.shape[0] - 1)
    pca   = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    meta_models = []
    meta_probs  = np.zeros_like(oof_probs)

    for i, cls in enumerate(CLASSES):
        X_meta = np.column_stack([oof_probs[:, i], X_pca])
        m = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        m.fit(X_meta, y[:, i])
        meta_models.append(m)
        meta_probs[:, i] = m.predict_proba(X_meta)[:, 1]

    meta_thr   = tune_thresholds(meta_probs, y)
    meta_preds = (meta_probs >= meta_thr).astype(int)
    print_metrics(y, meta_preds, label="[Meta in-sample — optimistic]")

    # 5. Final XGB on full training data (used at test time for level-1 features)
    print("\n[5/5] Training final XGB on full training data ...")
    xgb_models = train_xgb(X, y)

    # Return meta-model thresholds for submission
    return xgb_models, meta_models, pca, meta_thr


# =========================
# SUBMISSION
# =========================

def make_submission(mode, xgb_models, meta_models, pca, thresholds, out_file):
    """
    Generate submission CSV for the given mode.

    For each test peptide:
      1. Compute features (seq embedding + optional structure).
      2. Get XGB probability (level-1 feature).
      3. PCA-transform features using training-fitted PCA.
      4. Feed [xgb_prob, pca_features] to meta LogReg.
      5. Apply thresholds → binary predictions → write CSV.
    """
    print(f"\n[Submission] mode={mode}  output={out_file}")

    ids, X_test = load_test_data(mode)
    print(f"  Test samples: {len(ids)},  feature dim: {X_test.shape[1]}")

    xgb_probs = predict_xgb(xgb_models, X_test)   # (n_test, n_classes)

    print("\nSubmission probability diagnostics:")
    for i, cls in enumerate(CLASSES):
        print(
            f"{cls:30s} "
            f"mean_prob={xgb_probs[:, i].mean():.4f} "
            f"threshold={thresholds[i]:.4f}"
        )

    X_pca = pca.transform(X_test)                 # (n_test, n_components)

    rows = []
    for j, pid in enumerate(ids):
        preds = []
        for i in range(len(CLASSES)):
            x_meta = np.column_stack([[xgb_probs[j, i]], X_pca[j:j+1]])
            prob   = meta_models[i].predict_proba(x_meta)[0, 1]
            preds.append(int(prob >= thresholds[i]))
        rows.append([pid] + preds)

    sub_df = pd.DataFrame(rows, columns=["ID"] + CLASSES)
    sub_df.to_csv(out_file, index=False)
    print(f"  Saved {len(sub_df)} rows → {out_file}")
    return sub_df


# =========================
# SAVE / LOAD ARTIFACTS
# =========================

def save_artifacts(path, xgb_models, meta_models, pca, thresholds):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "xgb_models.pkl"),  "wb") as f: pickle.dump(xgb_models,  f)
    with open(os.path.join(path, "meta_models.pkl"), "wb") as f: pickle.dump(meta_models, f)
    with open(os.path.join(path, "pca.pkl"),         "wb") as f: pickle.dump(pca,         f)
    np.save(os.path.join(path, "thresholds.npy"), thresholds)
    print(f"  Artifacts saved → {path}/")


def load_artifacts(path):
    with open(os.path.join(path, "xgb_models.pkl"),  "rb") as f: xgb_models  = pickle.load(f)
    with open(os.path.join(path, "meta_models.pkl"), "rb") as f: meta_models = pickle.load(f)
    with open(os.path.join(path, "pca.pkl"),         "rb") as f: pca         = pickle.load(f)
    thresholds = np.load(os.path.join(path, "thresholds.npy"))
    return xgb_models, meta_models, pca, thresholds


# =========================
# MAIN
# =========================

def main():
    N_SPLITS = 3

    # ── Mode 1: Sequence only ──────────────────────────────
    xgb_seq, meta_seq, pca_seq, thr_seq = run_pipeline("seq_only", N_SPLITS)
    save_artifacts("artifacts/seq_only", xgb_seq, meta_seq, pca_seq, thr_seq)
    make_submission(
        mode="seq_only",
        xgb_models=xgb_seq,
        meta_models=meta_seq,
        pca=pca_seq,
        thresholds=thr_seq,
        out_file="submission_seq_only.csv",
    )

    # ── Mode 2: Sequence + Structure ──────────────────────
    xgb_str, meta_str, pca_str, thr_str = run_pipeline("seq_struct", N_SPLITS)
    save_artifacts("artifacts/seq_struct", xgb_str, meta_str, pca_str, thr_str)
    make_submission(
        mode="seq_struct",
        xgb_models=xgb_str,
        meta_models=meta_str,
        pca=pca_str,
        thresholds=thr_str,
        out_file="submission_seq_struct.csv",
    )

    print("\nDone.")
    print("  submission_seq_only.csv")
    print("  submission_seq_struct.csv")


if __name__ == "__main__":
    main()