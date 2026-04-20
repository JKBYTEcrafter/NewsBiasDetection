import sys
"""SVM Binary"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
df = pd.read_excel(sys.argv[1])

# 2. Extract features and labels
sentences = df.iloc[:, 0].astype(str).tolist()     # Hindi sentences
original_labels = df.iloc[:, 2].astype(str).tolist()  # Original labels

# 3. Merge 'Partially Legitimate' and 'Synthetic' into 'Synthetic'
binary_labels = ['Pure' if label == 'Legitimate' else 'Septic' for label in original_labels]

# 4. Encode binary labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(binary_labels)  # Legitimate=0, Synthetic=1

# 5. Generate sentence embeddings
model = SentenceTransformer('distiluse-base-multilingual-cased')
embeddings = model.encode(sentences)

# 6. Split data (use stratify to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# 7. Train SVM
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# 8. Predict
y_pred = svm_model.predict(X_test)

# 9. Decode for reporting
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# 10. Evaluate
print("SVM Accuracy (Legitimate vs Synthetic):", accuracy_score(y_test_labels, y_pred_labels))
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
