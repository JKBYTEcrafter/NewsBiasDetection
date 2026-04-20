import sys
"""ANN"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

df = pd.read_excel(sys.argv[1], header=None)

texts = df.iloc[:, 0].astype(str).tolist()
raw_labels = df.iloc[:, 2].astype(str).str.strip().str.lower().tolist()
binary_labels = []
for lbl in raw_labels:
    if lbl == "legitimate":
        binary_labels.append(0)   # Legitimate
    else:
        binary_labels.append(1)   # Synthetic
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    binary_labels,
    test_size=0.2,
    random_state=42,
    stratify=binary_labels
)
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2),
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_vec.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    X_train_vec,
    np.array(y_train),
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    verbose=1
)
y_pred = (model.predict(X_test_vec) > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}\n")

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
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(sys.argv[2] + '.png', bbox_inches='tight')
plt.close()
