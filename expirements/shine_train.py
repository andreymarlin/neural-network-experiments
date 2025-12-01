import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
import os
import json
from datetime import datetime
from tqdm.auto import tqdm

# ---------------------------------------------------
# FIX FREEZE ISSUES (very important)
# ---------------------------------------------------
os.environ["DISABLE_SAFETENSORS"] = "1"          # prevent slow safetensors download
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # avoid fork deadlocks

# ---------------------------------------------------
# MLflow
# ---------------------------------------------------
import mlflow
import mlflow.sklearn

os.environ["MLFLOW_TRACKING_USERNAME"] = "Shine76225"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "6adedd94750f26726d84f19f85977104fbd3e1c1"

mlflow.set_tracking_uri("https://dagshub.com/Shine76225/neural-network-experiments.mlflow")
mlflow.set_experiment("shine_experiment_dev2")

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
MODEL_NAME = "sentence-transformers/msmarco-MiniLM-L-12-v3"
EMBEDDING_DIM = 384

DATA_DIR = "/Users/shinekhantaung/Documents/dcs_project/neural-network-experiments/data"
TRAIN_FILE = f"{DATA_DIR}/train.csv"
VAL_FILE   = f"{DATA_DIR}/val.csv"
TEST_FILE  = f"{DATA_DIR}/test.csv"

OUTPUT_DIR = "shine_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n==============================")
print(" SHINE EXPERIMENT START")
print("==============================\n")

# ---------------------------------------------------
# ENABLE MPS GPU
# ---------------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
    print("⚡ Using Apple Silicon GPU (MPS)\n")
else:
    device = "cpu"
    print(" Using CPU\n")

# ---------------------------------------------------
# LOAD + FIX DATA
# ---------------------------------------------------
def load_and_fix(df, filename="UNKNOWN"):
    print("\n----------------------------------")
    print(f"🔍 Reading file: {filename}")
    print("Available columns:", list(df.columns))

    col_lower = {c.lower(): c for c in df.columns}
    s1_candidates = ["text", "sentence1", "original", "source"]
    s2_candidates = ["paraphrase", "sentence2", "target"]

    s1 = next((col_lower[n] for n in s1_candidates if n in col_lower), None)
    s2 = next((col_lower[n] for n in s2_candidates if n in col_lower), None)

    if s1 is None or s2 is None:
        raise ValueError(f"Column detection failed for {filename}")

    print(f"✔ Detected sentence1 = '{s1}'")
    print(f"✔ Detected sentence2 = '{s2}'")
    print("----------------------------------\n")

    df = df.rename(columns={s1: "sentence1", s2: "sentence2"})
    df["label"] = 1
    return df

print("📌 Loading datasets...")

train_df = load_and_fix(pd.read_csv(TRAIN_FILE, sep=None, engine="python"), TRAIN_FILE)
val_df   = load_and_fix(pd.read_csv(VAL_FILE,   sep=None, engine="python"), VAL_FILE)
test_df  = load_and_fix(pd.read_csv(TEST_FILE,  sep=None, engine="python"), TEST_FILE)

print(f"Train size: {len(train_df)}")
print(f"Val size:   {len(val_df)}")
print(f"Test size:  {len(test_df)}\n")

# ---------------------------------------------------
# LOAD MODEL (FIX SAFETENSORS FREEZE)
# ---------------------------------------------------
print(f"📌 Loading model: {MODEL_NAME}\n")
model = SentenceTransformer(MODEL_NAME)
model.to(device)

# ---------------------------------------------------
# EMBEDDING
# ---------------------------------------------------
def embed_fast(df, split, batch_size=256):
    print(f"\n📌 Embedding {split.upper()} (batch_size={batch_size})")

    s1 = df["sentence1"].tolist()
    s2 = df["sentence2"].tolist()
    all_s1, all_s2 = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc=f" {split} batches"):
            emb1 = model.encode(s1[i:i+batch_size], convert_to_tensor=True, device=device)
            emb2 = model.encode(s2[i:i+batch_size], convert_to_tensor=True, device=device)
            all_s1.append(emb1.cpu())
            all_s2.append(emb2.cpu())

    s1_final = torch.cat(all_s1)
    s2_final = torch.cat(all_s2)

    torch.save(s1_final, f"{OUTPUT_DIR}/{split}_embed_s1.pt")
    torch.save(s2_final, f"{OUTPUT_DIR}/{split}_embed_s2.pt")

    print(f"✔ Saved {split} embeddings → {OUTPUT_DIR}/")
    return s1_final, s2_final

train_s1, train_s2 = embed_fast(train_df, "train")
val_s1, val_s2     = embed_fast(val_df,   "val")
test_s1, test_s2   = embed_fast(test_df,  "test")

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
def compute_metrics(e1, e2):
    print("\n📌 Computing cosine similarity metrics...")

    cosine_scores = util.cos_sim(e1, e2).diag().numpy()
    threshold = 0.7

    preds = (cosine_scores >= threshold).astype(int)
    labels = np.ones(len(preds))

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "mean_cosine": float(np.mean(cosine_scores)),
        "median_cosine": float(np.median(cosine_scores)),
    }

train_metrics = compute_metrics(train_s1, train_s2)
val_metrics   = compute_metrics(val_s1, val_s2)
test_metrics  = compute_metrics(test_s1, test_s2)

# ---------------------------------------------------
# MLflow logging
# ---------------------------------------------------
with mlflow.start_run(run_name="shine_dev2_run"):
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("embedding_dim", EMBEDDING_DIM)
    mlflow.log_param("device", device)

    for k, v in train_metrics.items(): mlflow.log_metric(f"train_{k}", v)
    for k, v in val_metrics.items(): mlflow.log_metric(f"val_{k}", v)
    for k, v in test_metrics.items(): mlflow.log_metric(f"test_{k}", v)

    metrics_path = f"{OUTPUT_DIR}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics
        }, f, indent=4)

    mlflow.log_artifact(metrics_path)

# ---------------------------------------------------
# FINISH
# ---------------------------------------------------
print("\n==============================")
print("✔ Results saved → shine_outputs/metrics.json")
print("✔ SHINE EXPERIMENT FINISHED!")
print("==============================\n")
