import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
df = pd.read_excel(sys.argv[1], header=None)

texts = df.iloc[:, 0].astype(str).tolist()
raw_labels = df.iloc[:, 2].astype(str).str.strip().str.lower().tolist()

# Binary labels
labels = [0 if l == "legitimate" else 1 for l in raw_labels]
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,3),
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)

lr_probs = lr.predict_proba(X_test_vec)[:, 1]
X_train_dense = X_train_vec.toarray()
X_test_dense = X_test_vec.toarray()

ann = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_dense.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

ann.compile(
    optimizer=Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ann.fit(
    X_train_dense,
    np.array(y_train),
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

ann_probs = ann.predict(X_test_dense).flatten()
# Average probabilities
ensemble_probs = (lr_probs + ann_probs) / 2

# Final predictions
y_pred = (ensemble_probs > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nEnsemble Accuracy: {accuracy:.4f}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Synthetic"]))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Legitimate", "Synthetic"],
    yticklabels=["Legitimate", "Synthetic"],
    cmap="Greens"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Ensemble Confusion Matrix")
plt.savefig(sys.argv[2] + '.png', bbox_inches='tight')
plt.close()
