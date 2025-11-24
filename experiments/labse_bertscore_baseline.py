import pandas as pd
from bert_score import score
import mlflow
from sentence_transformers import SentenceTransformer

# ---- config ----
MODEL_NAME = "cointegrated/LaBSE-en-ru"
EMBEDDING_DIM = 768

TRAIN_PATH = "/Users/ask/NSU/class/DSC/neural-network-experiments/train.csv"
VAL_PATH = "/Users/ask/NSU/class/DSC/neural-network-experiments/val.csv"
TEST_PATH = "/Users/ask/NSU/class/DSC/neural-network-experiments/test.csv"

# ---- (optional) load your assigned model, so this experiment is tied to LaBSE ----
print(f"Loading sentence transformer model: {MODEL_NAME}")
st_model = SentenceTransformer(MODEL_NAME)
print("Model loaded. Embedding dim (expected):", EMBEDDING_DIM)

# ---- load data ----
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)
print("Test shape:", test_df.shape)
print("Columns:", test_df.columns)

# We only need the texts for BERTScore:
refs  = list(test_df["sentence2"])
cands = list(test_df["sentence1"])

# (Optional) limit for speed during first run
MAX_SAMPLES = 2000  # you can increase later
refs  = refs[:MAX_SAMPLES]
cands = cands[:MAX_SAMPLES]

print(f"Computing BERTScore on {len(cands)} pairs...")

# ---- compute BERTScore ----
P, R, F1 = score(cands, refs, lang="en")
bert_f1 = F1.mean().item()
print("BERTScore_F1:", bert_f1)

# ---- log to MLflow ----
mlflow.set_experiment("ahsan_labse_tapaco")

with mlflow.start_run():
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("embedding_dim", EMBEDDING_DIM)
    mlflow.log_param("max_samples", MAX_SAMPLES)
    mlflow.log_metric("BERTScore_F1", bert_f1)

print("Logged to MLflow.")
