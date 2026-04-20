import sys
"""Binary NB"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
file_path = sys.argv[1]
df = pd.read_excel(file_path)

# 2. Extract Hindi text and original labels
X = df.iloc[:, 0].astype(str)  # Hindi text column
original_labels = df.iloc[:, 2].astype(str)  # Label column

# 3. Combine 'Partially Legitimate' and 'Synthetic' into 'Synthetic'
binary_labels = ['Legitimate' if label == 'Legitimate' else 'Synthetic' for label in original_labels]

# 4. Encode binary labels (Legitimate = 0, Synthetic = 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(binary_labels)

# 5. Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# 6. Split dataset (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)

# 7. Train Naïve Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# 8. Predict on test set
y_pred = nb_model.predict(X_test)

# 9. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Naïve Bayes Model Accuracy (Legitimate vs Synthetic): {accuracy:.2f}")

# Optional: Detailed classification report
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Confusion Matrix ===
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")

plt.savefig(sys.argv[2] + '.png', bbox_inches='tight')
plt.close()
