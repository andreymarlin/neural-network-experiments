import pandas as pd
from bert_score import score
import mlflow
from sentence_transformers import SentenceTransformer
import os

# 设置环境变量强制离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# ---- config ----
SENTENCE_TRANSFORMER_PATH = "/Users/hushuai/Desktop/neural-network-experiments/paraphrase-multilingual-mpnet-base-v2"
ROBERTA_MODEL_PATH = "/Users/hushuai/Desktop/neural-network-experiments/roberta-large"  # 替换为你的 RoBERTa-large 本地路径
EMBEDDING_DIM = 768

TRAIN_PATH = "../data/train.csv"
VAL_PATH = "../data/val.csv"
TEST_PATH = "../data/test.csv"

# ---- 加载 SentenceTransformer 模型 ----
print(f"Loading sentence transformer model from: {SENTENCE_TRANSFORMER_PATH}")
st_model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)
print("✓ SentenceTransformer model loaded successfully")

# ---- 验证 RoBERTa 模型路径 ----
print(f"Checking RoBERTa model at: {ROBERTA_MODEL_PATH}")
if not os.path.exists(ROBERTA_MODEL_PATH):
    print(f"错误: RoBERTa 模型路径不存在: {ROBERTA_MODEL_PATH}")
    exit(1)

required_files = ['config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt']
for file in required_files:
    file_path = os.path.join(ROBERTA_MODEL_PATH, file)
    if os.path.exists(file_path):
        print(f"✓ {file}")
    else:
        print(f"⚠ {file} - 可能缺失")

# ---- load data ----
print("\nLoading data...")
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)
print("Test shape:", test_df.shape)

refs  = list(test_df["sentence2"])
cands = list(test_df["sentence1"])

MAX_SAMPLES = 2000
refs  = refs[:MAX_SAMPLES]
cands = cands[:MAX_SAMPLES]

print(f"Computing BERTScore on {len(cands)} pairs using local RoBERTa model...")

# ---- 使用本地 RoBERTa 模型计算 BERTScore ----
try:
    P, R, F1 = score(
        cands, 
        refs, 
        model_type=ROBERTA_MODEL_PATH,  # 使用本地路径
        lang="en",
        num_layers=17,  # RoBERTa-large 使用第17层（共24层）
        verbose=True,
        idf=False,
        batch_size=32,
        device="cpu"  # 使用CPU
    )
    
    bert_f1 = F1.mean().item()
    bert_precision = P.mean().item()
    bert_recall = R.mean().item()
    
    print(f"\n=== BERTScore Results ===")
    print(f"Precision: {bert_precision:.4f}")
    print(f"Recall:    {bert_recall:.4f}")
    print(f"F1:        {bert_f1:.4f}")
    
except Exception as e:
    print(f"BERTScore 计算失败: {e}")
    
    # 备选方案：尝试不同的层数
    try:
        print("尝试使用不同的层数...")
        P, R, F1 = score(
            cands, 
            refs, 
            model_type=ROBERTA_MODEL_PATH,
            lang="en",
            num_layers=13,  # 尝试中间层
            verbose=True,
            idf=False
        )
        
        bert_f1 = F1.mean().item()
        bert_precision = P.mean().item()
        bert_recall = R.mean().item()
        
        print(f"\n=== BERTScore Results (Layer 13) ===")
        print(f"Precision: {bert_precision:.4f}")
        print(f"Recall:    {bert_recall:.4f}")
        print(f"F1:        {bert_f1:.4f}")
        
    except Exception as e2:
        print(f"所有 BERTScore 尝试都失败: {e2}")
        exit(1)

import os
os.environ['MLFLOW_TRACKING_USERNAME'] = 'tutuhuss'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '81330f25b06ab6b447a9989cc88b18d293b8420b'
# ---- log to MLflow ----
mlflow.set_tracking_uri("https://dagshub.com/andreymarlin/neural-network-experiments.mlflow")
mlflow.set_experiment("hushuai_mpnetv2_tapaco")

with mlflow.start_run(run_name="mpnetv2_baseline"):
    mlflow.log_param("sentence_transformer_model", "paraphrase-multilingual-mpnet-base-v2")
    mlflow.log_param("bertscore_model", "roberta-large-local")
    mlflow.log_param("max_samples", MAX_SAMPLES)
    mlflow.log_param("embedding_dim", EMBEDDING_DIM)
    
    mlflow.log_metric("BERTScore_F1", bert_f1)
    mlflow.log_metric("BERTScore_P", bert_precision)
    mlflow.log_metric("BERTScore_R", bert_recall)

print("\n✓ Successfully logged to MLflow")


