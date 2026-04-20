import sys
import os
import re
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_and_export():
    # Resolve exact absolute paths relative to where train.py is physically located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "..", "Dataset For Research Paper.xlsx")
    print(f"Loading dataset from {dataset_path}...")
    
    try:
        df = pd.read_excel(dataset_path)
    except FileNotFoundError:
        print(f"Error: Could not find {dataset_path}. Please provide correct path!")
        sys.exit(1)

    print("Cleaning data...")
    texts = df.iloc[:, 0].astype(str).tolist()
    # Updated to column 1 since 'text' is 0 and 'label' is 1
    original_labels = df.iloc[:, 1].astype(str).tolist()

    # Consolidate labels exactly as built in research architecture
    binary_labels = ['Legitimate' if label == 'Legitimate' else 'Synthetic' for label in original_labels]
    texts_cleaned = [clean_text(t) for t in texts]

    print("Loading SentenceTransformer model... (this will download model if not cached)")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print(f"Generating embeddings for {len(texts_cleaned)} records...")
    X_embeddings = embedding_model.encode(texts_cleaned, show_progress_bar=True)
    
    print("Training LogisticRegressionCV...")
    clf = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000, n_jobs=-1)
    clf.fit(X_embeddings, binary_labels)
    
    print("Evaluating Model Accuracy on Training set:")
    y_pred = clf.predict(X_embeddings)
    print(f"Accuracy: {accuracy_score(binary_labels, y_pred) * 100:.2f}%")
    print(classification_report(binary_labels, y_pred))

    model_path = os.path.join(script_dir, "model", "lr_model.pkl")
    print(f"Saving LogicRegression model locally to '{model_path}'...")
    joblib.dump(clf, model_path)
    
    print("Export Complete! You can now run the web application.")

if __name__ == "__main__":
    train_and_export()
