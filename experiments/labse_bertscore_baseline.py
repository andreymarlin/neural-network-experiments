from pathlib import Path
import shutil
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from bert_score import score
import mlflow
from sentence_transformers import SentenceTransformer
import torch

# =====================
# SETTINGS
# =====================

# Toggle: do we upload the BIG model zip to DagsHub?
# - False  -> only metrics + tiny summary artifact (fast, recommended for daily use)
# - True   -> also upload labse_ahsan_baseline.zip (~hundreds of MB, slow)
LOG_MODEL_TO_DAGSHUB = True

# =====================
# Device detection
# =====================

if torch.backends.mps.is_available():
    DEVICE = "mps"       # Apple GPU on M-series
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)

# =====================
# Config
# =====================

MODEL_NAME = "cointegrated/LaBSE-en-ru"
EMBEDDING_DIM = 768

MAX_SAMPLES = 2000
BERT_BATCH_SIZE = 128

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "tapaco_paraphrases_dataset.csv"
TRAIN_PATH = PROJECT_ROOT / "train.csv"
VAL_PATH = PROJECT_ROOT / "val.csv"
TEST_PATH = PROJECT_ROOT / "test.csv"

MLFLOW_EXPERIMENT_NAME = "ahsan_labse_tapaco_v2"

MODEL_DIR = PROJECT_ROOT / "models" / "labse_ahsan_baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# Step 1: Prepare data
# =====================

def prepare_splits():
    if TRAIN_PATH.exists() and VAL_PATH.exists() and TEST_PATH.exists():
        print("Found existing train/val/test CSVs. Skipping split.")
        return

    print(f"Reading raw data from {RAW_DATA_PATH} ...")
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep="\t",
        engine="python",
        on_bad_lines="skip",
    )

    print("Original columns:", df.columns)
    print("Total rows:", len(df))
    print(df.head())

    text_col, para_col = df.columns[:2]
    df = df[[text_col, para_col]].rename(
        columns={text_col: "sentence1", para_col: "sentence2"}
    )

    print("\nRenamed columns:", df.columns)
    print(df.head())

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    print(f"\nSplit sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"\nSaved: {TRAIN_PATH.name}, {VAL_PATH.name}, {TEST_PATH.name}")


# =====================
# Step 2: Load & save LaBSE
# =====================

def load_and_save_model():
    print(f"\nLoading sentence transformer model: {MODEL_NAME}")
    st_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print("Model loaded on device:", DEVICE)

    print(f"Saving model to {MODEL_DIR} ...")
    st_model.save(str(MODEL_DIR))
    print("Model saved locally.")

    # Prepare a zip locally (only uploaded if LOG_MODEL_TO_DAGSHUB = True)
    zip_path = PROJECT_ROOT / "models" / "labse_ahsan_baseline"
    zip_file = shutil.make_archive(str(zip_path), "zip", root_dir=str(MODEL_DIR))
    print(f"Created local model archive: {zip_file}")

    return st_model, Path(zip_file)


# =====================
# Step 3: Compute BERTScore
# =====================

def compute_bertscore():
    print("\nLoading split data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print("Columns:", test_df.columns)

    refs = list(test_df["sentence2"])
    cands = list(test_df["sentence1"])

    if MAX_SAMPLES is not None:
        refs = refs[:MAX_SAMPLES]
        cands = cands[:MAX_SAMPLES]

    num_pairs = len(cands)
    print(f"\nComputing BERTScore on {num_pairs} pairs "
          f"(batch_size={BERT_BATCH_SIZE}, device={DEVICE})...")

    P, R, F1 = score(
        cands,
        refs,
        lang="en",
        device=DEVICE,
        batch_size=BERT_BATCH_SIZE,
        verbose=True,
    )

    P = P.detach().cpu()
    R = R.detach().cpu()
    F1 = F1.detach().cpu()

    P_mean = P.mean().item()
    R_mean = R.mean().item()
    F1_mean = F1.mean().item()
    F1_std = F1.std(unbiased=False).item()

    print(f"\nBERTScore_P:      {P_mean}")
    print(f"BERTScore_R:      {R_mean}")
    print(f"BERTScore_F1:     {F1_mean}")
    print(f"BERTScore_F1_std: {F1_std}")

    return (P_mean, R_mean, F1_mean, F1_std,
            train_df, val_df, test_df, num_pairs)


# =====================
# Step 4: Log to MLflow
# =====================

def log_to_mlflow(P_mean, R_mean, F1_mean, F1_std,
                  train_df, val_df, test_df, num_pairs,
                  model_zip_path: Path):
    print("\nLogging to MLflow...")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="labse_ahsan_full_pipeline"):
        # Params
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("embedding_dim", EMBEDDING_DIM)
        mlflow.log_param("device", DEVICE)
        mlflow.log_param("max_samples", MAX_SAMPLES)
        mlflow.log_param("bert_batch_size", BERT_BATCH_SIZE)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("val_rows", len(val_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("num_pairs_eval", num_pairs)
        mlflow.log_param("log_model_to_dagshub", LOG_MODEL_TO_DAGSHUB)

        # Metrics
        mlflow.log_metric("BERTScore_P", P_mean)
        mlflow.log_metric("BERTScore_R", R_mean)
        mlflow.log_metric("BERTScore_F1", F1_mean)
        mlflow.log_metric("BERTScore_F1_std", F1_std)

        # ---- Artifacts ----

        # Always: small summary artifact (fast)
        summary_path = PROJECT_ROOT / "models" / "run_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Device: {DEVICE}\n")
            f.write(f"Pairs evaluated: {num_pairs}\n")
            f.write(f"BERTScore_F1: {F1_mean}\n")

        t1 = time.time()
        mlflow.log_artifact(str(summary_path), artifact_path="run_info")
        t_summary = time.time() - t1
        print(f"Summary artifact upload finished in {t_summary:.2f} seconds.")

        # Optional: big model zip upload
        if LOG_MODEL_TO_DAGSHUB:
            size_mb = model_zip_path.stat().st_size / (1024 * 1024)
            print(f"\nUploading model artifact: {model_zip_path.name} "
                  f"({size_mb:.1f} MB) to MLflow/DagsHub...")

            t0 = time.time()
            mlflow.log_artifact(str(model_zip_path), artifact_path="model")
            t_upload = time.time() - t0
            print(f"Model artifact upload finished in {t_upload:.1f} seconds.")
        else:
            print("\nLOG_MODEL_TO_DAGSHUB is False → skipping big model upload.")

    print("Logged metrics and artifacts to MLflow.")


# =====================
# Main
# =====================

if __name__ == "__main__":
    overall_start = time.time()

    prepare_splits()

    _, model_zip_path = load_and_save_model()

    (P_mean, R_mean, F1_mean, F1_std,
     train_df, val_df, test_df, num_pairs) = compute_bertscore()

    log_to_mlflow(P_mean, R_mean, F1_mean, F1_std,
                  train_df, val_df, test_df, num_pairs,
                  model_zip_path)

    total_sec = time.time() - overall_start
    print(f"\nPipeline finished   (total time: {total_sec:.1f} seconds)")
