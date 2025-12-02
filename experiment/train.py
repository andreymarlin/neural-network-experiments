import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import mlflow
import os
import re  
import random
from tqdm import tqdm
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

SENTENCE_TRANSFORMER_PATH = "/work/dvc/paraphrase-multilingual-mpnet-base-v2"
ROBERTA_MODEL_PATH = "/work/dvc/roberta-large"
EMBEDDING_DIM = 768

TRAIN_PATH = "../data/train.csv"
VAL_PATH = "../data/val.csv"
TEST_PATH = "../data/test.csv"

TRAIN_CONFIG = {
    "batch_size": 8,  
    "gradient_accumulation_step":4,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "max_seq_length": 64,  
    "evaluation_steps": 200,
    "output_dir": "./contrastive_finetuned_model",
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.01,
    "fp16": False
}

def sanitize_mlflow_dict(config_dict):
    cleaned = {}
    for key, value in config_dict.items():
        str_key = str(key)
        safe_key = re.sub(r'[^a-zA-Z0-9_\.\- ]', '', str_key).replace(' ', '_')
        safe_key = safe_key[:250]
        str_value = str(value)
        safe_value = str_value[:500]
        cleaned[safe_key] = safe_value
    return cleaned

class PositivePairDataset(Dataset):
    def __init__(self, csv_path, max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df[:max_samples]
        
        print(f"Columns in {csv_path}: {list(self.df.columns)}")
        
        col_names = self.df.columns.tolist()
        if len(col_names) >= 2:
            self.sentences1 = self.df[col_names[0]].astype(str).tolist()
            self.sentences2 = self.df[col_names[1]].astype(str).tolist()
        else:
            raise ValueError(f"CSV at least need 2 col，only have {len(col_names)}")
        
        print(f"Loaded {len(self.sentences1)} pairs from {csv_path}")
    
    def __len__(self):
        return len(self.sentences1)
    
    def __getitem__(self, idx):
        return {
            "sentence1": self.sentences1[idx],
            "sentence2": self.sentences2[idx],
        }

print("=" * 60)
print("Setting up training with Sentence Transformers...")
print("=" * 60)

train_dataset = PositivePairDataset(TRAIN_PATH)
val_dataset = PositivePairDataset(VAL_PATH)
test_dataset = PositivePairDataset(TEST_PATH)

print(f"\nDataset sizes:")
print(f"  Training: {len(train_dataset)} pairs")
print(f"  Validation: {len(val_dataset)} pairs")
print(f"  Test: {len(test_dataset)} pairs")

train_examples = []
for i in range(len(train_dataset)):
    item = train_dataset[i]
    train_examples.append(InputExample(
        texts=[item['sentence1'], item['sentence2']],
        label=1.0
    ))

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

print(f"\nLoading model from: {SENTENCE_TRANSFORMER_PATH}")
#model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH,device='cpu')
model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)
model.max_seq_length = TRAIN_CONFIG["max_seq_length"]
#model.to('cpu')
print("\nSetting up MultipleNegativesRankingLoss...")
train_loss = losses.MultipleNegativesRankingLoss(model=model)

print("\nCreating data loader...")
train_dataloader = DataLoader(
    train_examples,
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=True,
    num_workers=0
)

print("Creating evaluator...")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    val_examples,
    name='val_similarity',
    show_progress_bar=True
)

warmup_steps = int(len(train_dataloader) * TRAIN_CONFIG["num_epochs"] * TRAIN_CONFIG["warmup_ratio"])
print(f"\nTraining configuration:")
print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
print(f"  Total steps: {len(train_dataloader) * TRAIN_CONFIG['num_epochs']}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Learning rate: {TRAIN_CONFIG['learning_rate']}")


os.environ['MLFLOW_TRACKING_USERNAME'] = 'tutuhuss'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'n1ceshuai@'

mlflow.set_tracking_uri("https://dagshub.com/andreymarlin/neural-network-experiments.mlflow")
mlflow.set_experiment("hushuai_mpnetv2_tapaco")

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)
mlflow.autolog(disable=True) 
with mlflow.start_run(run_name=f"mpnetv2_fintune_1"):
    cleaned_config = sanitize_mlflow_dict(TRAIN_CONFIG)  

    mlflow.log_params(cleaned_config)  
    
    mlflow.log_param("base_model", "paraphrase-multilingual-mpnet-base-v2")
    mlflow.log_param("train_samples", len(train_examples))
    mlflow.log_param("val_samples", len(val_examples))
    mlflow.log_param("loss_function", "MultipleNegativesRankingLoss")
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=TRAIN_CONFIG["num_epochs"],
            evaluation_steps=TRAIN_CONFIG["evaluation_steps"],
            warmup_steps=warmup_steps,
            output_path=TRAIN_CONFIG["output_dir"],
            save_best_model=True,
            show_progress_bar=True,
            use_amp=False,  
            optimizer_params={
                'lr': TRAIN_CONFIG["learning_rate"], 
                'eps': 1e-6  
            }
        )
        
        print("✓ Training completed!")
        
    except Exception as e:
        print(f"Training error: {e}")
        print("\nTrying alternative approach...")
        
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
            
            TRAIN_CONFIG["batch_size"] = 2  
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
                    'lr': TRAIN_CONFIG["learning_rate"] * 0.5,  
                }
            )
            print("✓ Training completed with batch size 2!")
    
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    test_examples = []
    test_size = min(2000, len(test_dataset))
    for i in range(test_size):
        item = test_dataset[i]
        test_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=1.0
        ))
    
    model.eval()
    all_similarities = []
    all_labels = []
    
    batch_size = 32
    for i in range(0, len(test_examples), batch_size):
        batch_examples = test_examples[i:i+batch_size]
        
        sentences1 = [ex.texts[0] for ex in batch_examples]
        sentences2 = [ex.texts[1] for ex in batch_examples]
        labels = [ex.label for ex in batch_examples]
        
        embeddings1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=False)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=False)
        
        similarities = F.cosine_similarity(embeddings1, embeddings2)
        
        all_similarities.extend(similarities.cpu().numpy())
        all_labels.extend(labels)
    
    avg_similarity = np.mean(all_similarities)
    print(f"\nTest set results:")
    print(f"  Average similarity: {avg_similarity:.4f}")
    print(f"  Std deviation: {np.std(all_similarities):.4f}")
    print(f"  Min similarity: {np.min(all_similarities):.4f}")
    print(f"  Max similarity: {np.max(all_similarities):.4f}")
    
    mlflow.log_metric("test_avg_similarity", avg_similarity)
    
    try:
        from bert_score import score
        
        print("\n" + "=" * 60)
        print("Computing BERTScore...")
        print("=" * 60)
        
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
        
        mlflow.log_metric("BERTScore_F1", bert_f1)
        mlflow.log_metric("BERTScore_P", bert_precision)
        mlflow.log_metric("BERTScore_R", bert_recall)
        
    except Exception as e:
        print(f"BERTScore evaluation failed: {e}")
    
    final_model_path = os.path.join(TRAIN_CONFIG["output_dir"], "final_model")
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    mlflow.pytorch.log_model(model, "model")
    
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
