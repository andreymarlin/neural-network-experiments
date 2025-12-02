import os
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from bert_score import score
from sklearn.model_selection import train_test_split
import mlflow
import dagshub
import pickle
import json
from datetime import datetime

# Initialize DAGsHub and MLflow - Pointing to collaborative repository
dagshub.init(repo_owner='andreymarlin', 
             repo_name='neural-network-experiments', 
             mlflow=True)
mlflow.set_experiment("experiment/dev-5")

# Configuration
MODEL_NAME = "sentence-transformers/distilbert-base-nli-mean-tokens"
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
MODEL_DIR = "./models/tapaco_distilbert"
DATA_DIR = "./data"
ARTIFACTS_DIR = "./artifacts"
RAW_DATA_PATH = os.path.join(DATA_DIR, "tapaco_paraphrases_dataset.csv")

def ensure_directories():
    """Create necessary directories"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def save_model_artifacts(model, train_df, test_df, metrics):
    """Save model and artifacts locally and prepare for DVC tracking"""
    
    # 1. Save the trained model (already saved by sentence-transformers in MODEL_DIR)
    
    # 2. Save model as pickle file for easy loading
    model_pickle_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    with open(model_pickle_path, 'wb') as f:
        pickle.dump(model, f)
    
    # 3. Save training metadata
    metadata = {
        "model_name": MODEL_NAME,
        "training_date": datetime.now().isoformat(),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "metrics": metrics
    }
    
    metadata_path = os.path.join(ARTIFACTS_DIR, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 4. Save model configuration (FIXED)
    config = {
        "model_config": {
            "model_name": MODEL_NAME,
            "embedding_dimension": model.get_sentence_embedding_dimension(),
            "max_seq_length": model.max_seq_length,
            "device": str(model.device)
        },
        "training_params": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR
        }
    }
    
    config_path = os.path.join(ARTIFACTS_DIR, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Model artifacts saved locally!")
    return model_pickle_path, metadata_path, config_path

def prepare_data():
    """Prepare and split the dataset"""
    print("📊 Preparing dataset...")
    
    # Load the dataset
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep="\t",                  # Tab-separated
        names=["sentence_1", "sentence_2"],  # Column names
        header=0,
        engine="python"
    )

    # Clean the dataset
    df = df.dropna(subset=["sentence_1", "sentence_2"])
    df = df.drop_duplicates()
    df["label"] = 1.0  # All pairs are paraphrases

    print("Dataset loaded and cleaned:")
    print(df.head())
    print(f"Total rows: {len(df)}")

    # Split into train / val / test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save the splits
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    print("✅ Dataset split complete!")
    print(f"Train: {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"Test: {test_df.shape}")
    
    return train_df, val_df, test_df

def train_model():
    """Train the sentence transformer model"""
    print("🚀 Starting model training...")
    
    # Load training data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    
    # Prepare training examples
    train_examples = [
        InputExample(texts=[row['sentence_1'], row['sentence_2']], label=float(row['label']))
        for _, row in train_df.iterrows()
    ]
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    model = SentenceTransformer(MODEL_NAME)
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Train model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=100,
        optimizer_params={'lr': LR},
        show_progress_bar=True
    )
    
    # Save trained model
    model.save(MODEL_DIR)
    print("✅ Model training complete!")
    
    return model

def generate_predictions(model):
    """Generate predictions on test data"""
    print("📊 Generating predictions...")
    
    # Load test data
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Generate embeddings and compute similarity
    def compute_similarity(row):
        emb1 = model.encode(row['sentence_1'])
        emb2 = model.encode(row['sentence_2'])
        cosine_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        return cosine_sim
    
    test_df['similarity'] = test_df.apply(compute_similarity, axis=1)
    
    # Save predictions
    predictions_path = os.path.join(DATA_DIR, "test_predictions.csv")
    test_df.to_csv(predictions_path, index=False)
    print("✅ Predictions saved!")
    
    return predictions_path

def evaluate_predictions(predictions_path):
    """Evaluate predictions using BERTScore"""
    print("📈 Evaluating predictions...")
    
    test_df = pd.read_csv(predictions_path)
    
    # BERTScore evaluation
    P, R, F1 = score(
        test_df['sentence_1'].tolist(),
        test_df['sentence_2'].tolist(),
        lang="en",
        model_type="distilbert-base-uncased",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add BERTScore columns to dataframe
    test_df['BERTScore_F1'] = F1.tolist()
    test_df['BERTScore_Precision'] = P.tolist()
    test_df['BERTScore_Recall'] = R.tolist()
    
    # Save evaluation results
    eval_path = os.path.join(DATA_DIR, "test_predictions_eval.csv")
    test_df.to_csv(eval_path, index=False)
    print("✅ Evaluation complete!")
    
    metrics = {
        "f1_mean": F1.mean().item(),
        "p_mean": P.mean().item(),
        "r_mean": R.mean().item()
    }
    
    return eval_path, metrics

def main():
    """Main execution function"""
    ensure_directories()
    
    with mlflow.start_run(run_name="tapaco-distilbert-full-pipeline"):
        # Log parameters
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "train_test_split_random_state": 42,
            "test_size": 0.2,
            "val_size": 0.1  # 0.2 * 0.5 = 0.1 of original
        })
        
        # Step 1: Prepare data
        train_df, val_df, test_df = prepare_data()
        
        # Log dataset information
        mlflow.log_params({
            "total_samples": len(train_df) + len(val_df) + len(test_df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df)
        })
        
        # Log data artifacts
        mlflow.log_artifacts(DATA_DIR, artifact_path="data")
        
        # Step 2: Train model
        model = train_model()
        mlflow.log_artifacts(MODEL_DIR, artifact_path="model")
        
        # Step 3: Generate predictions
        predictions_path = generate_predictions(model)
        mlflow.log_artifact(predictions_path, artifact_path="predictions")
        
        # Step 4: Evaluate predictions
        eval_path, metrics = evaluate_predictions(predictions_path)
        
        # Log metrics
        mlflow.log_metrics({
            "BERTScore_F1_test": metrics["f1_mean"],
            "BERTScore_Precision_test": metrics["p_mean"],
            "BERTScore_Recall_test": metrics["r_mean"]
        })
        
        # Log evaluation artifacts
        mlflow.log_artifact(eval_path, artifact_path="evaluation")
        
        # Step 5: Save additional model artifacts
        model_pickle_path, metadata_path, config_path = save_model_artifacts(
            model, train_df, test_df, metrics
        )
        
        # Log additional artifacts to MLflow
        mlflow.log_artifact(model_pickle_path, artifact_path="artifacts")
        mlflow.log_artifact(metadata_path, artifact_path="artifacts")
        mlflow.log_artifact(config_path, artifact_path="artifacts")
        mlflow.log_artifacts(ARTIFACTS_DIR, artifact_path="artifacts")
        
        print("🎉 Full pipeline complete! All results logged to MLflow.")
        print(f"📊 Final Metrics:")
        print(f"   BERTScore F1: {metrics['f1_mean']:.4f}")
        print(f"   BERTScore Precision: {metrics['p_mean']:.4f}")
        print(f"   BERTScore Recall: {metrics['r_mean']:.4f}")
        print(f"📁 Artifacts saved in: {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()