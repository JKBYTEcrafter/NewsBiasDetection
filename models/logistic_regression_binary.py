import sys
"""Binary Logistic Regression"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_excel(sys.argv[1])  # Change path if needed

# 2. Extract text and labels
texts = df.iloc[:, 0].astype(str).tolist()
original_labels = df.iloc[:, 2].astype(str).tolist()

# 3. Combine 'Partially Legitimate' and 'Synthetic' into one class
binary_labels = ['Pure' if label == 'Legitimate' else 'Septic' for label in original_labels]

# 4. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(binary_labels)  # Legitimate = 0, Synthetic = 1

# 5. Encode text using SentenceTransformer
model = SentenceTransformer('distiluse-base-multilingual-cased')  # Fast multilingual embedding
X = model.encode(texts, show_progress_bar=True)

# 6. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 7. Train Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# 8. Predict and evaluate
y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (Legitimate vs Synthetic): {accuracy:.4f}")

# Optional: detailed classification report
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)

# Print confusion matrix as array
print("\nConfusion Matrix (raw values):\n", cm)

# Display as a nice plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(sys.argv[2] + '.png', bbox_inches='tight')
plt.close()
