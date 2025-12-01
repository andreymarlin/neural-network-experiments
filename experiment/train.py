import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import mlflow
import os
import re  # 修改点：导入正则表达式库用于参数清理
import random
from tqdm import tqdm
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置环境变量强制离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
# 修改点：禁用tokenizer并行性以避免警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ---- config ----
SENTENCE_TRANSFORMER_PATH = "/work/dvc/paraphrase-multilingual-mpnet-base-v2"
ROBERTA_MODEL_PATH = "/work/dvc/roberta-large"
EMBEDDING_DIM = 768

TRAIN_PATH = "../data/train.csv"
VAL_PATH = "../data/val.csv"
TEST_PATH = "../data/test.csv"

# 训练配置 - 修改点：进一步减小配置以缓解内存压力
TRAIN_CONFIG = {
    "batch_size": 8,  
    "gradient_accumulation_step":4,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "max_seq_length": 64,  # 修改点：从128减小到64，显著降低内存占用
    "evaluation_steps": 200,
    "output_dir": "./contrastive_finetuned_model",
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.01,
    "fp16": False
}

# ---- MLflow参数清理函数（新增）----
def sanitize_mlflow_dict(config_dict):
    """清理字典，确保所有键和值都是字符串，并移除非字母数字字符。"""
    cleaned = {}
    for key, value in config_dict.items():
        # 确保键为字符串，并清理
        str_key = str(key)
        # 移除非字母数字、下划线、点、横杠和空格之外的字符，并将空格替换为下划线
        safe_key = re.sub(r'[^a-zA-Z0-9_\.\- ]', '', str_key).replace(' ', '_')
        # 可选：截断过长的键名
        safe_key = safe_key[:250]
        # 确保值为字符串，并截断过长值
        str_value = str(value)
        safe_value = str_value[:500]
        cleaned[safe_key] = safe_value
    return cleaned

# ---- 1. 数据集类 ----
class PositivePairDataset(Dataset):
    def __init__(self, csv_path, max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df[:max_samples]
        
        # 检查列名
        print(f"Columns in {csv_path}: {list(self.df.columns)}")
        
        # 尝试不同的列名组合
        col_names = self.df.columns.tolist()
        if len(col_names) >= 2:
            self.sentences1 = self.df[col_names[0]].astype(str).tolist()
            self.sentences2 = self.df[col_names[1]].astype(str).tolist()
        else:
            raise ValueError(f"CSV文件需要至少2列，只有{len(col_names)}列")
        
        print(f"Loaded {len(self.sentences1)} pairs from {csv_path}")
    
    def __len__(self):
        return len(self.sentences1)
    
    def __getitem__(self, idx):
        return {
            "sentence1": self.sentences1[idx],
            "sentence2": self.sentences2[idx],
        }

# ---- 2. 主程序 ----
print("=" * 60)
print("Setting up training with Sentence Transformers...")
print("=" * 60)

# 加载数据集
train_dataset = PositivePairDataset(TRAIN_PATH)
val_dataset = PositivePairDataset(VAL_PATH)
test_dataset = PositivePairDataset(TEST_PATH)

print(f"\nDataset sizes:")
print(f"  Training: {len(train_dataset)} pairs")
print(f"  Validation: {len(val_dataset)} pairs")
print(f"  Test: {len(test_dataset)} pairs")

# 创建训练样本
train_examples = []
for i in range(len(train_dataset)):
    item = train_dataset[i]
    train_examples.append(InputExample(
        texts=[item['sentence1'], item['sentence2']],
        label=1.0
    ))

# 创建验证样本（限制数量）
val_examples = []
val_size = min(1000, len(val_dataset))
for i in range(val_size):
    item = val_dataset[i]
    val_examples.append(InputExample(
        texts=[item['sentence1'], item['sentence2']],
        label=1.0
    ))

print(f"\nCreated {len(train_examples)} training examples")
print(f"Created {len(val_examples)} validation examples")

# 加载模型
print(f"\nLoading model from: {SENTENCE_TRANSFORMER_PATH}")
#model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH,device='cpu')
model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)
model.max_seq_length = TRAIN_CONFIG["max_seq_length"]
#model.to('cpu')
# 设置损失函数
print("\nSetting up MultipleNegativesRankingLoss...")
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# 创建训练数据加载器 - 使用简单的DataLoader
print("\nCreating data loader...")
train_dataloader = DataLoader(
    train_examples,
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=True,
    num_workers=0
)

# 创建评估器
print("Creating evaluator...")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    val_examples,
    name='val_similarity',
    show_progress_bar=True
)

# 计算warmup steps
warmup_steps = int(len(train_dataloader) * TRAIN_CONFIG["num_epochs"] * TRAIN_CONFIG["warmup_ratio"])
print(f"\nTraining configuration:")
print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
print(f"  Total steps: {len(train_dataloader) * TRAIN_CONFIG['num_epochs']}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Learning rate: {TRAIN_CONFIG['learning_rate']}")


os.environ['MLFLOW_TRACKING_USERNAME'] = 'tutuhuss'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'n1ceshuai@'

# ---- log to MLflow ----
mlflow.set_tracking_uri("https://dagshub.com/andreymarlin/neural-network-experiments.mlflow")
mlflow.set_experiment("hushuai_mpnetv2_tapaco")

# ---- 4. 训练模型 ----
print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)
mlflow.autolog(disable=True) 
with mlflow.start_run(run_name=f"mpnetv2_fintune_1"):
    # 记录配置 - 修改点：使用清理后的参数
    cleaned_config = sanitize_mlflow_dict(TRAIN_CONFIG)  # 清理配置字典

    mlflow.log_params(cleaned_config)  # 记录清理后的参数
    
    mlflow.log_param("base_model", "paraphrase-multilingual-mpnet-base-v2")
    mlflow.log_param("train_samples", len(train_examples))
    mlflow.log_param("val_samples", len(val_examples))
    mlflow.log_param("loss_function", "MultipleNegativesRankingLoss")
    
    try:
        # 训练模型 - 修改点：已移除 'correct_bias' 参数
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=TRAIN_CONFIG["num_epochs"],
            evaluation_steps=TRAIN_CONFIG["evaluation_steps"],
            warmup_steps=warmup_steps,
            output_path=TRAIN_CONFIG["output_dir"],
            save_best_model=True,
            show_progress_bar=True,
            use_amp=False,  # Mac上关闭混合精度
            optimizer_params={
                'lr': TRAIN_CONFIG["learning_rate"], 
                'eps': 1e-6  # 修改点：已移除 'correct_bias'
            }
        )
        
        print("✓ Training completed!")
        
    except Exception as e:
        print(f"Training error: {e}")
        print("\nTrying alternative approach...")
        
        # 尝试使用更简单的配置
        try:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=TRAIN_CONFIG["num_epochs"],
                warmup_steps=warmup_steps,
                output_path=TRAIN_CONFIG["output_dir"],
                show_progress_bar=True,
                use_amp=False,
                optimizer_params={
                    'lr': TRAIN_CONFIG["learning_rate"], 
                    'eps': 1e-6
                }
            )
            print("✓ Training completed with simpler configuration!")
        except Exception as e2:
            print(f"Still failing: {e2}")
            print("\nTrying with even smaller batch size...")
            
            # 使用更小的batch size
            TRAIN_CONFIG["batch_size"] = 2  # 修改点：进一步减小到2
            train_dataloader = DataLoader(
                train_examples,
                batch_size=TRAIN_CONFIG["batch_size"],
                shuffle=True,
                num_workers=0
            )
            
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=TRAIN_CONFIG["num_epochs"],
                warmup_steps=warmup_steps // 2,
                output_path=TRAIN_CONFIG["output_dir"],
                show_progress_bar=True,
                use_amp=False,
                optimizer_params={
                    'lr': TRAIN_CONFIG["learning_rate"] * 0.5,  # 更小的学习率
                }
            )
            print("✓ Training completed with batch size 2!")
    
    # ---- 5. 在测试集上评估 ----
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    # 创建测试样本
    test_examples = []
    test_size = min(2000, len(test_dataset))
    for i in range(test_size):
        item = test_dataset[i]
        test_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=1.0
        ))
    
    # 计算测试集上的相似度
    model.eval()
    all_similarities = []
    all_labels = []
    
    # 分批处理避免内存问题
    batch_size = 32
    for i in range(0, len(test_examples), batch_size):
        batch_examples = test_examples[i:i+batch_size]
        
        sentences1 = [ex.texts[0] for ex in batch_examples]
        sentences2 = [ex.texts[1] for ex in batch_examples]
        labels = [ex.label for ex in batch_examples]
        
        # 生成嵌入
        embeddings1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=False)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=False)
        
        # 计算余弦相似度
        similarities = F.cosine_similarity(embeddings1, embeddings2)
        
        all_similarities.extend(similarities.cpu().numpy())
        all_labels.extend(labels)
    
    # 计算平均相似度
    avg_similarity = np.mean(all_similarities)
    print(f"\nTest set results:")
    print(f"  Average similarity: {avg_similarity:.4f}")
    print(f"  Std deviation: {np.std(all_similarities):.4f}")
    print(f"  Min similarity: {np.min(all_similarities):.4f}")
    print(f"  Max similarity: {np.max(all_similarities):.4f}")
    
    # 记录测试分数
    mlflow.log_metric("test_avg_similarity", avg_similarity)
    
    # ---- 6. 计算BERTScore ----
    try:
        from bert_score import score
        
        print("\n" + "=" * 60)
        print("Computing BERTScore...")
        print("=" * 60)
        
        # 准备测试数据（使用前500个样本）
        test_sentences1 = [ex.texts[0] for ex in test_examples[:500]]
        test_sentences2 = [ex.texts[1] for ex in test_examples[:500]]
        
        print(f"Computing BERTScore on {len(test_sentences1)} pairs...")
        
        P, R, F1 = score(
            test_sentences1, 
            test_sentences2, 
            model_type=ROBERTA_MODEL_PATH,
            lang="en",
            num_layers=17,
            verbose=True,
            idf=False,
            batch_size=16,
            device="cpu"
        )
        
        bert_f1 = F1.mean().item()
        bert_precision = P.mean().item()
        bert_recall = R.mean().item()
        
        print(f"\n=== BERTScore Results ===")
        print(f"Precision: {bert_precision:.4f}")
        print(f"Recall:    {bert_recall:.4f}")
        print(f"F1:        {bert_f1:.4f}")
        
        # 记录到MLflow
        mlflow.log_metric("BERTScore_F1", bert_f1)
        mlflow.log_metric("BERTScore_P", bert_precision)
        mlflow.log_metric("BERTScore_R", bert_recall)
        
    except Exception as e:
        print(f"BERTScore evaluation failed: {e}")
    
    # ---- 7. 保存最终模型 ----
    final_model_path = os.path.join(TRAIN_CONFIG["output_dir"], "final_model")
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # 保存到MLflow
    mlflow.pytorch.log_model(model, "model")
    
    # ---- 8. 示例展示 ----
    print("\n" + "=" * 60)
    print("Example Inference")
    print("=" * 60)
    
    examples = [
        ("The cat sits on the mat", "A cat is sitting on a mat"),
        ("The weather is nice today", "Today has beautiful weather"),
        ("I love machine learning", "Deep learning is fascinating"),
        ("He is cooking dinner", "She prepared the evening meal"),
        ("The company released new products", "New products were launched"),
    ]
    
    print("\nCosine similarities between sentence pairs:")
    print("-" * 80)
    
    for sent1, sent2 in examples:
        embeddings = model.encode([sent1, sent2], convert_to_tensor=True)
        similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
        
        print(f"\n\"{sent1}\"")
        print(f"\"{sent2}\"")
        print(f"  → Similarity: {similarity:.4f}")
        
        # 判断是否相似
        if similarity > 0.7:
            print(f"  → Prediction: Similar (✓)")
        elif similarity > 0.5:
            print(f"  → Prediction: Somewhat similar (~)")
        else:
            print(f"  → Prediction: Not similar (✗)")

print("\n" + "=" * 60)
print("Training completed successfully!")
print("=" * 60)
print(f"Model saved to: {TRAIN_CONFIG['output_dir']}")
print(f"Test average similarity: {avg_similarity:.4f}")
print("=" * 60)