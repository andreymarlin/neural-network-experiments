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
# CONFIG
# ---------------------------------------------------

MODEL_NAME = "sentence-transformers/msmarco-MiniLM-L-12-v3"
EMBEDDING_DIM = 384

DATA_DIR = "/Users/shinekhantaung/Documents/dcs_project/neural-network-experiments/data"
TRAIN_FILE = f"{DATA_DIR}/train.csv"
VAL_FILE = f"{DATA_DIR}/val.csv"
TEST_FILE = f"{DATA_DIR}/test.csv"

OUTPUT_DIR = "shine_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n==============================")
print(" SHINE EXPERIMENT START")
print("==============================\n")

# ---------------------------------------------------
# ENABLE APPLE SILICON GPU (MPS)
# ---------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
    print("⚡ Using Apple Silicon GPU (MPS)\n")
else:
    device = "cpu"
    print("🖥 Using CPU\n")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

print(" Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE, sep="\t", header=0)
val_df = pd.read_csv(VAL_FILE, sep="\t", header=0)
test_df = pd.read_csv(TEST_FILE, sep="\t", header=0)

train_df = train_df.rename(columns={"Text": "sentence1", "Paraphrase": "sentence2"})
val_df = val_df.rename(columns={"Text": "sentence1", "Paraphrase": "sentence2"})
test_df = test_df.rename(columns={"Text": "sentence1", "Paraphrase": "sentence2"})

train_df["label"] = 1
val_df["label"] = 1
test_df["label"] = 1

print(f"Train size: {len(train_df)} rows")
print(f"Val size:   {len(val_df)} rows")
print(f"Test size:  {len(test_df)} rows\n")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

print(f"Loading model: {MODEL_NAME}\n")
model = SentenceTransformer(MODEL_NAME)
model.to(device)

# ---------------------------------------------------
# OPTIMIZED EMBEDDING (BATCHED + GPU + TQDM)
# ---------------------------------------------------

def embed_fast(df, split, batch_size=256):
    print(f"\n Embedding {split.upper()} (batch_size={batch_size})")
    
    sentences1 = df["sentence1"].tolist()
    sentences2 = df["sentence2"].tolist()

    all_s1 = []
    all_s2 = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc=f" {split} batches"):
            batch_s1 = sentences1[i:i+batch_size]
            batch_s2 = sentences2[i:i+batch_size]

            emb1 = model.encode(batch_s1, convert_to_tensor=True, device=device)
            emb2 = model.encode(batch_s2, convert_to_tensor=True, device=device)

            all_s1.append(emb1.cpu())
            all_s2.append(emb2.cpu())

    s1_final = torch.cat(all_s1)
    s2_final = torch.cat(all_s2)

    torch.save(s1_final, f"{OUTPUT_DIR}/{split}_embed_s1.pt")
    torch.save(s2_final, f"{OUTPUT_DIR}/{split}_embed_s2.pt")

    print(f"Saved {split} embeddings → {OUTPUT_DIR}/")
    return s1_final, s2_final


train_s1, train_s2 = embed_fast(train_df, "train")
val_s1, val_s2 = embed_fast(val_df, "val")
test_s1, test_s2 = embed_fast(test_df, "test")

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

def compute_metrics(e1, e2):
    print("\n Computing cosine similarity metrics...")
    cosine_scores = util.cos_sim(e1, e2).diag().numpy()

    threshold = 0.7
    labels = np.ones(len(cosine_scores))
    preds = (cosine_scores >= threshold).astype(int)

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
val_metrics = compute_metrics(val_s1, val_s2)
test_metrics = compute_metrics(test_s1, test_s2)

# ---------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------

results = {
    "model": MODEL_NAME,
    "device": device,
    "embedding_dim": EMBEDDING_DIM,
    "timestamp": datetime.now().isoformat(),
    "train": train_metrics,
    "val": val_metrics,
    "test": test_metrics,
}

with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n==============================")
print(" Results saved → shine_outputs/metrics.json")
print(" SHINE EXPERIMENT FINISHED!")
print("==============================\n")
