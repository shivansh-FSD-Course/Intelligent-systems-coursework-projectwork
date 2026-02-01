import os
import time
import tempfile
import numpy as np
import pandas as pd
import re

DATA_PATH = os.path.expanduser("~/Desktop/data")

try:
    import fasttext
    print(" fasttext loaded")
except ImportError:
    print("  fasttext not found!")
    print("  Install with: pip install fasttext")
    exit(1)

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

#Load data

def load_datasets(data_path):
    
    train_path = os.path.join(data_path, "Train.csv")
    valid_path = os.path.join(data_path, "Valid.csv")
    test_path = os.path.join(data_path, "Test.csv")
    
    try:
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)
        
        return train_df, valid_df, test_df
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None

#Text processing

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_fasttext_file(df, filepath):
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            text = clean_text(row['text'])
            label = row['label']
            # FastText format: __label__0 or __label__1 followed by text
            f.write(f"__label__{label} {text}\n")
    return filepath

#Fast text training

def evaluate_model(model, valid_file):
    """Evaluate model on validation set."""
    # FastText's test function returns (num_samples, precision, recall)
    result = model.test(valid_file)
    # For binary classification, precision@1 equals accuracy
    return result[1]  # precision (which equals accuracy for single-label)

def get_predictions(model, texts):
    predictions = []
    for text in texts:
        pred = model.predict(clean_text(text))
        # pred[0] is like ('__label__1',), extract the label
        label = int(pred[0][0].replace('__label__', ''))
        predictions.append(label)
    return np.array(predictions)

def train_fasttext_with_tuning(train_file, valid_file):
    # Define hyperparameter search space
    # These are reasonable ranges based on FastText documentation
    param_grid = {
        'lr': [0.1, 0.5, 1.0],           # Learning rate
        'epoch': [5, 10, 25],             # Number of epochs
        'wordNgrams': [1, 2, 3],          # Word n-grams
        'dim': [50, 100],                 # Vector dimensions (keep small for Mac)
    }
    
    # Fixed parameters
    fixed_params = {
        'ws': 5,              # Context window size
        'minCount': 1,        # Min word frequency
        'minn': 0,            # Min char n-gram (0 = no char n-grams for speed)
        'maxn': 0,            # Max char n-gram
        'loss': 'softmax',    # Loss function
        'thread': 4,          # Number of threads
    }
    print(f"    Learning rates: {param_grid['lr']}")
    print(f"    Epochs: {param_grid['epoch']}")
    print(f"    Word n-grams: {param_grid['wordNgrams']}")
    print(f"    Dimensions: {param_grid['dim']}")
    
    total_combinations = (len(param_grid['lr']) * len(param_grid['epoch']) * 
                         len(param_grid['wordNgrams']) * len(param_grid['dim']))
    print(f"\n  Total configurations to test: {total_combinations}")
    print("\n  Training and evaluating...")
    
    best_accuracy = 0
    best_params = {}
    best_model = None
    results = []
    
    start_time = time.time()
    count = 0
    
    for lr in param_grid['lr']:
        for epoch in param_grid['epoch']:
            for wordNgrams in param_grid['wordNgrams']:
                for dim in param_grid['dim']:
                    count += 1
                    
                    # Train model with current hyperparameters
                    model = fasttext.train_supervised(
                        input=train_file,
                        lr=lr,
                        epoch=epoch,
                        wordNgrams=wordNgrams,
                        dim=dim,
                        **fixed_params,
                        verbose=0  # Suppress output
                    )
                    
                    # Evaluate on validation set
                    accuracy = evaluate_model(model, valid_file)
                    
                    results.append({
                        'lr': lr,
                        'epoch': epoch,
                        'wordNgrams': wordNgrams,
                        'dim': dim,
                        'accuracy': accuracy
                    })
                    
                    # Update best
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'lr': lr,
                            'epoch': epoch,
                            'wordNgrams': wordNgrams,
                            'dim': dim
                        }
                        best_model = model
                    
                    # Progress
                    print(f"  [{count:2}/{total_combinations}] lr={lr}, epoch={epoch:2}, "
                          f"ngrams={wordNgrams}, dim={dim:3} → Acc: {accuracy:.4f}"
                          f"{'  ★ BEST' if accuracy == best_accuracy else ''}")
    
    elapsed = time.time() - start_time
    print(f"\n  Best Configuration:")
    print(f"    Learning rate:  {best_params['lr']}")
    print(f"    Epochs:         {best_params['epoch']}")
    print(f"    Word n-grams:   {best_params['wordNgrams']}")
    print(f"    Dimensions:     {best_params['dim']}")
    print(f"    Valid Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    return best_model, best_params, best_accuracy, pd.DataFrame(results)

#Finlan Eval

def final_evaluation(model, test_df, test_file):
    
    # Get predictions
    test_texts = test_df['text'].values
    test_labels = test_df['label'].values
    
    predictions = get_predictions(model, test_texts)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    
    print(f"\n  TEST SET RESULTS:")
    print(f"  -----------------")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Neg    Pos")
    print(f"  Actual Neg  {cm[0,0]:5}  {cm[0,1]:5}")
    print(f"  Actual Pos  {cm[1,0]:5}  {cm[1,1]:5}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#main

def main():
    total_start = time.time()
    
    # Load data
    train_df, valid_df, test_df = load_datasets(DATA_PATH)
    
    if train_df is None:
        print("\n Failed to load data")
        return
    
    # Use temp directory for FastText format files
    temp_dir = tempfile.mkdtemp()
    train_file = os.path.join(temp_dir, "train.txt")
    valid_file = os.path.join(temp_dir, "valid.txt")
    test_file = os.path.join(temp_dir, "test.txt")
    
    prepare_fasttext_file(train_df, train_file)
    prepare_fasttext_file(valid_df, valid_file)
    prepare_fasttext_file(test_df, test_file)
    
    # Train with hyperparameter tuning
    best_model, best_params, valid_accuracy, results_df = train_fasttext_with_tuning(
        train_file, valid_file
    )
    
    # Save hyperparameter search results
    results_path = os.path.join(DATA_PATH, "fasttext_hyperparam_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Final evaluation on test set
    test_results = final_evaluation(best_model, test_df, test_file)
    
    # Save model
    model_path = os.path.join(DATA_PATH, "fasttext_sentiment_model.bin")
    best_model.save_model(model_path)
    print(f"\n Model saved to: {model_path}")
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    # Summary
    total_time = time.time() - total_start
    print("SUMMARY")
    print(f"\n  Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  Validation Accuracy: {valid_accuracy:.4f}")
    print(f"  Test Accuracy:       {test_results['accuracy']:.4f}")
    print(f"  Test F1-Score:       {test_results['f1']:.4f}")
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

if __name__ == "__main__":
    main()
