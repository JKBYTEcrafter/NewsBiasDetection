import sys
# --- Import Required Libraries ---
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV  # Using CV version for auto-tuning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer  # --- NEW LIBRARY ---

# 1️⃣ Load the dataset
# Make sure this path is correct for your environment
df = pd.read_excel(sys.argv[1])

# 2️⃣ Extract text and labels
texts = df.iloc[:, 0].astype(str).tolist()
original_labels = df.iloc[:, 2].astype(str).tolist()

# 3️⃣ Combine 'Partially Legitimate' + 'Synthetic' into one class
binary_labels = ['Legitimate' if label == 'Legitimate' else 'Synthetic' for label in original_labels]

# 4️⃣ Hindi text cleaning (Simpler cleaning is often better for Transformers)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # Just normalize whitespace
    return text

texts_cleaned = [clean_text(t) for t in texts]

# 5️⃣ Train-test split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts_cleaned, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels
)

# 6️⃣ --- NEW: Generate Sentence Embeddings ---
# We use a model pre-trained on many languages, including Hindi.
# This model converts each sentence into a 768-dimension vector.
print("Loading sentence transformer model (this may take a minute)...")

# --- THIS IS THE CORRECTED LINE ---
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Generating embeddings for training data...")
X_train_embeddings = model.encode(X_train_text, show_progress_bar=True)

print("Generating embeddings for test data...")
X_test_embeddings = model.encode(X_test_text, show_progress_bar=True)

print("\nEmbeddings generated. Shape:", X_train_embeddings.shape)

# 7️⃣ --- NEW: Train a Classifier on the Embeddings ---
# Now that we have dense vectors (embeddings) instead of sparse TF-IDF,
# we can train a simple, strong classifier on them.
print("Training classifier on embeddings...")
# LogisticRegressionCV automatically finds the best 'C' parameter
clf = LogisticRegressionCV(
    cv=5,
    random_state=42,
    max_iter=1000,
    n_jobs=-1
)
clf.fit(X_train_embeddings, y_train)

# 8️⃣ Predictions and Evaluation
print("Training complete. Evaluating on test set...")
y_pred = clf.predict(X_test_embeddings)

accuracy = accuracy_score(y_test, y_pred)
print("✅ Final Test Accuracy (Embeddings):", round(accuracy * 100, 2), "%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 🧩 Confusion Matrix + Heatmap
cm = confusion_matrix(y_test, y_pred, labels=['Legitimate', 'Synthetic'])
print("🧩 Confusion Matrix:\n", cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Synthetic'], yticklabels=['Legitimate', 'Synthetic'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Sentence Embeddings)')
plt.savefig(sys.argv[2] + '.png', bbox_inches='tight')
plt.close()
