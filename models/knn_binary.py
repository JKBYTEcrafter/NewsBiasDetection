import sys
"""Binary KNN"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# === Load the dataset ===
file_path = sys.argv[1]  # Change this to your actual path
df = pd.read_excel(file_path)

# === Extract Hindi text and original labels ===
X = df.iloc[:, 0].astype(str)  # Ensure it's all strings
y_original = df.iloc[:, 2].astype(str)

# === Merge 'Partially Legitimate' and 'Synthetic' into 'Synthetic' ===
y_binary = ['Legitimate' if label == 'Legitimate' else 'Synthetic' for label in y_original]

# === Encode binary labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_binary)  # Legitimate = 0, Synthetic = 1

# === Convert text to TF-IDF vectors ===
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# === Train the KNN classifier ===
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# === Make predictions ===
y_pred = knn.predict(X_test)

# === Decode predictions to labels (optional) ===
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# === Evaluate the model ===
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy (Legitimate vs Synthetic): {accuracy:.2f}")

# === Optional: Show class-wise performance ===
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
