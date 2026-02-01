import os
import time
import warnings
import numpy as np
import pandas as pd
import re

warnings.filterwarnings('ignore')

DATA_PATH = os.path.expanduser("~/Desktop/data")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)

# Stop words
try:
    from stop_words import get_stop_words
    STOP_WORDS_LIST = get_stop_words('en')
except ImportError:
    exit(1)

# Gensim for FastText
try:
    import gensim.downloader as gensim_api
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

#Load Data

def load_datasets(data_path):
    
    train_path = os.path.join(data_path, "Train.csv")
    valid_path = os.path.join(data_path, "Valid.csv")
    test_path = os.path.join(data_path, "Test.csv")
    
    try:
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)
        
        print(f" Train: {len(train_df):,} samples")
        print(f" Valid: {len(valid_df):,} samples")
        print(f" Test:  {len(test_df):,} samples")
        
        # Display column names and sample
        print(f"\n  Columns: {list(train_df.columns)}")
        print(f"  Label distribution (train): {dict(train_df['label'].value_counts())}")
        
        return train_df, valid_df, test_df
        
    except FileNotFoundError as e:
        print(f" Error: {e}")
        print(f" Available files in {data_path}:")
        if os.path.exists(data_path):
            for f in os.listdir(data_path):
                print(f"    - {f}")
        return None, None, None

#Process Text

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

def preprocess_data(train_df, valid_df, test_df):
    
    start_time = time.time()
    
    # Clean text
    train_texts = train_df['text'].apply(clean_text).values
    valid_texts = valid_df['text'].apply(clean_text).values
    test_texts = test_df['text'].apply(clean_text).values
    
    # Labels
    train_labels = train_df['label'].values
    valid_labels = valid_df['label'].values
    test_labels = test_df['label'].values
    
    elapsed = time.time() - start_time
    print(f"complete ({elapsed:.1f}s)")
    
    return (train_texts, train_labels), (valid_texts, valid_labels), (test_texts, test_labels)

#TF IDF Without stop words reoval

def approach1_tfidf_no_stopwords(train_data, valid_data, test_data):
    
    train_texts, train_labels = train_data
    valid_texts, valid_labels = valid_data
    test_texts, test_labels = test_data
    
    tfidf = TfidfVectorizer(
        max_features=10000,      # Limit due to computation power
        ngram_range=(1, 2),      # Unigrams and bigrams
        min_df=5,                # Minimum document frequency
        max_df=0.95,             # Maximum document frequency
        sublinear_tf=True,       # Apply sublinear tf scaling
        stop_words=None          # NO stop word removal
    )
    
    X_train = tfidf.fit_transform(train_texts)
    X_valid = tfidf.transform(valid_texts)
    X_test = tfidf.transform(test_texts)
    
    print(f" Feature matrix shape: {X_train.shape}")
    print(f"Vocab size: {len(tfidf.vocabulary_):,}")
    
    # Train Logistic Regression
    print("\n  Training Logistic Regression classifier...")
    
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, train_labels)
    
    # Validation results
    valid_pred = model.predict(X_valid)
    valid_acc = accuracy_score(valid_labels, valid_pred)
    valid_f1 = f1_score(valid_labels, valid_pred)
    
    print(f"\n  VALIDATION RESULTS:")
    print(f"  Accuracy:  {valid_acc:.4f} ({valid_acc*100:.2f}%)")
    print(f"  F1-Score:  {valid_f1:.4f}")
    print(f"  Precision: {precision_score(valid_labels, valid_pred):.4f}")
    print(f"  Recall:    {recall_score(valid_labels, valid_pred):.4f}")
    
    # Store for final evaluation
    return {
        'name': 'TF-IDF (No Stop Words)',
        'model': model,
        'vectorizer': tfidf,
        'X_test': X_test,
        'test_labels': test_labels,
        'valid_accuracy': valid_acc,
        'valid_f1': valid_f1
    }
#TF IDF With stop words removal

def approach2_tfidf_with_stopwords(train_data, valid_data, test_data):
    train_texts, train_labels = train_data
    valid_texts, valid_labels = valid_data
    test_texts, test_labels = test_data
    
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
        stop_words=STOP_WORDS_LIST  # USE stop word removal
    )
    
    X_train = tfidf.fit_transform(train_texts)
    X_valid = tfidf.transform(valid_texts)
    X_test = tfidf.transform(test_texts)
    
    print(f" Feature matrix shape: {X_train.shape}")
    print(f"Vocab size: {len(tfidf.vocabulary_):,}")
    
    # Train Logistic Regression
    print("\n  Training Logistic Regression classifier...")
    
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, train_labels)
  
    # Validation results
    valid_pred = model.predict(X_valid)
    valid_acc = accuracy_score(valid_labels, valid_pred)
    valid_f1 = f1_score(valid_labels, valid_pred)
    
    print(f"\n  VALIDATION RESULTS:")
    print(f"  -------------------")
    print(f"  Accuracy:  {valid_acc:.4f} ({valid_acc*100:.2f}%)")
    print(f"  F1-Score:  {valid_f1:.4f}")
    print(f"  Precision: {precision_score(valid_labels, valid_pred):.4f}")
    print(f"  Recall:    {recall_score(valid_labels, valid_pred):.4f}")
    
    return {
        'name': 'TF-IDF (With Stop Words Removed)',
        'model': model,
        'vectorizer': tfidf,
        'X_test': X_test,
        'test_labels': test_labels,
        'valid_accuracy': valid_acc,
        'valid_f1': valid_f1
    }

#Fast Text

def get_document_vector(text, ft_model, dim=300):
 
    words = text.split()
    word_vectors = []
    
    for word in words:
        try:
            vec = ft_model[word]
            word_vectors.append(vec)
        except KeyError:
            continue  # Skip OOV words
    
    if len(word_vectors) == 0:
        return np.zeros(dim)
    
    return np.mean(word_vectors, axis=0)

def approach3_fasttext(train_data, valid_data, test_data):
    
    if not GENSIM_AVAILABLE:
        print("\n  Gensim not available. Please download")
        print("  Install with: pip install gensim")
        return None
    
    train_texts, train_labels = train_data
    valid_texts, valid_labels = valid_data
    test_texts, test_labels = test_data
    
    # Load pre-trained FastText model
    print("\n  Loading pre-trained FastText model...")
    
    start_time = time.time()
    
    try:
        # This downloads the best pre-trained FastText model for English
        # It's trained on Wikipedia 2017, UMBC webbase, and statmt.org news
        ft_model = gensim_api.load('fasttext-wiki-news-subwords-300')
        print(f"  Model loaded ({time.time() - start_time:.1f}s)")
        print(f"  Vocab size: {len(ft_model):,} words")
        print(f"  Vector dimensions: {ft_model.vector_size}")
    except Exception as e:
        print(f"  Error loading model: {e}")
        print("\n  Trying alternative: word2vec-google-news-300...")
        try:
            ft_model = gensim_api.load('word2vec-google-news-300')
            print(f"   Alternative model loaded")
        except Exception as e2:
            print(f" Could not load any embedding model: {e2}")
            return None
    
    # Convert texts to document vectors
    print("\n  Converting texts to document vectors...")
    start_time = time.time()
    
    dim = ft_model.vector_size
    
    # Process in batches for memory efficiency
    def texts_to_vectors(texts, desc="Processing"):
        vectors = []
        total = len(texts)
        for i, text in enumerate(texts):
            if i % 5000 == 0:
                print(f"    {desc}: {i:,}/{total:,}")
            vectors.append(get_document_vector(text, ft_model, dim))
        return np.array(vectors)
    
    X_train = texts_to_vectors(train_texts, "Train")
    X_valid = texts_to_vectors(valid_texts, "Valid")
    X_test = texts_to_vectors(test_texts, "Test")
    
    print(f" Vectorization complete ({time.time() - start_time:.1f}s)")
    print(f" Feature matrix shape: {X_train.shape}")
        
    # Test a few C values (regularization strength)
    # Higher C = less regularization, lower C = more regularization
    c_values = [0.01, 0.1, 1.0, 10.0]
    best_c = 1.0
    best_valid_acc = 0
    
    for c in c_values:
        model_temp = LogisticRegression(
            C=c, 
            max_iter=1000, 
            solver='lbfgs',
            n_jobs=-1,
            random_state=42
        )
        model_temp.fit(X_train, train_labels)
        valid_pred_temp = model_temp.predict(X_valid)
        acc_temp = accuracy_score(valid_labels, valid_pred_temp)
        print(f"    C={c:5}: Validation Accuracy = {acc_temp:.4f}")
        
        if acc_temp > best_valid_acc:
            best_valid_acc = acc_temp
            best_c = c
    
    print(f"\n Best C value: {best_c} (Validation Accuracy: {best_valid_acc:.4f})")
    
    # Train final model with best hyperparameters
    print("\n  Training final model with best hyperparameters...")
    start_time = time.time()
    
    model = LogisticRegression(
        C=best_c,
        max_iter=1000,
        solver='lbfgs',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, train_labels)
    elapsed = time.time() - start_time
    print(f"Training complete ({elapsed:.1f}s)")
    
    # Validation results
    valid_pred = model.predict(X_valid)
    valid_acc = accuracy_score(valid_labels, valid_pred)
    valid_f1 = f1_score(valid_labels, valid_pred)
    
    print(f"\n  VALIDATION RESULTS:")
    print(f"  -------------------")
    print(f"  Accuracy:  {valid_acc:.4f} ({valid_acc*100:.2f}%)")
    print(f"  F1-Score:  {valid_f1:.4f}")
    print(f"  Precision: {precision_score(valid_labels, valid_pred):.4f}")
    print(f"  Recall:    {recall_score(valid_labels, valid_pred):.4f}")
    
    return {
        'name': 'FastText Embeddings',
        'model': model,
        'ft_model': ft_model,
        'X_test': X_test,
        'test_labels': test_labels,
        'valid_accuracy': valid_acc,
        'valid_f1': valid_f1,
        'best_C': best_c
    }

#Final Eval

def final_evaluation(results_list):
    
    print("FINAL EVALUATION ON TEST SET")
    
    final_results = []
    
    for result in results_list:
        if result is None:
            continue
            
        name = result['name']
        model = result['model']
        X_test = result['X_test']
        test_labels = result['test_labels']
        
        # Predict on test set
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        test_acc = accuracy_score(test_labels, test_pred)
        test_f1 = f1_score(test_labels, test_pred)
        test_prec = precision_score(test_labels, test_pred)
        test_rec = recall_score(test_labels, test_pred)
        
        print(f"\n  {name}")
        print(f"  {'-'*len(name)}")
        print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Test F1-Score:  {test_f1:.4f}")
        print(f"  Test Precision: {test_prec:.4f}")
        print(f"  Test Recall:    {test_rec:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(test_labels, test_pred)
        print(f"\n  Confusion Matrix:")
        print(f"                Predicted")
        print(f"              Neg    Pos")
        print(f"  Actual Neg  {cm[0,0]:5}  {cm[0,1]:5}")
        print(f"  Actual Pos  {cm[1,0]:5}  {cm[1,1]:5}")
        
        final_results.append({
            'Approach': name,
            'Valid_Accuracy': result['valid_accuracy'],
            'Valid_F1': result['valid_f1'],
            'Test_Accuracy': test_acc,
            'Test_F1': test_f1,
            'Test_Precision': test_prec,
            'Test_Recall': test_rec
        })
    
    return final_results

# =============================================================================
# COMPARISON SUMMARY
# =============================================================================

def print_comparison_summary(final_results):
    
    # Create DataFrame for nice display
    df = pd.DataFrame(final_results)
    
    print("\n  VALIDATION SET PERFORMANCE:")
    print("  " + "-"*60)
    for _, row in df.iterrows():
        print(f"  {row['Approach']:35} Acc: {row['Valid_Accuracy']:.4f}  F1: {row['Valid_F1']:.4f}")
    
    print("\n  TEST SET PERFORMANCE (Final):")
    print("  " + "-"*60)
    for _, row in df.iterrows():
        print(f"  {row['Approach']:35} Acc: {row['Test_Accuracy']:.4f}  F1: {row['Test_F1']:.4f}")
    
    # Find best approach
    best_idx = df['Test_Accuracy'].idxmax()
    best = df.loc[best_idx]
    print(f"\n  ★ BEST APPROACH: {best['Approach']}")
    print(f"    Test Accuracy: {best['Test_Accuracy']:.4f} ({best['Test_Accuracy']*100:.2f}%)")
    
    return df

#main

def main():
    
    total_start = time.time()
    
    # Load data
    train_df, valid_df, test_df = load_datasets(DATA_PATH)
    
    if train_df is None:
        print("\n Failed to load data.")
    
    # Preprocess
    train_data, valid_data, test_data = preprocess_data(train_df, valid_df, test_df)
    
    results = []
    
    # Approach 1: TF-IDF without stop words
    result1 = approach1_tfidf_no_stopwords(train_data, valid_data, test_data)
    results.append(result1)
    
    # Approach 2: TF-IDF with stop words
    result2 = approach2_tfidf_with_stopwords(train_data, valid_data, test_data)
    results.append(result2)
    
    # Approach 3: FastText
    result3 = approach3_fasttext(train_data, valid_data, test_data)
    results.append(result3)
    
    # Final evaluation on test set
    final_results = final_evaluation(results)
    
    # Print comparison
    comparison_df = print_comparison_summary(final_results)
    
    # Save results to CSV
    output_path = os.path.join(DATA_PATH, "sentiment_results.csv")
    comparison_df.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")
    
    total_time = time.time() - total_start
    print(f"time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main()
