import sys
"""LSTM"""

# -----------------------------------------------------------
# 1️⃣ IMPORTS
# -----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# -----------------------------------------------------------
# 2️⃣ LOAD EXCEL (NO HEADERS)
# -----------------------------------------------------------
df = pd.read_excel(sys.argv[1], header=None)

# Column 0 → Hindi sentence
sentences = df.iloc[:, 0].astype(str).tolist()

# Column 2 → Legitimate / Partially Legitimate / Synthetic
original_labels = df.iloc[:, 2].astype(str).str.strip().str.lower().tolist()

# -----------------------------------------------------------
# 3️⃣ CLEAN LABELS
# Combine Partially Legitimate + Synthetic → Synthetic
# -----------------------------------------------------------
cleaned = []
for label in original_labels:
    if label == "legitimate":
        cleaned.append("Legitimate")
    else:
        cleaned.append("Synthetic")

labels = cleaned

print("Class distribution:\n", pd.Series(labels).value_counts())

# -----------------------------------------------------------
# 4️⃣ ENCODE LABELS → 0/1
# -----------------------------------------------------------
label_to_int = {"Legitimate": 0, "Synthetic": 1}
y = np.array([label_to_int[l] for l in labels])

# -----------------------------------------------------------
# 5️⃣ TRAIN–TEST SPLIT
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------
# 6️⃣ TOKENIZATION (Hindi supported)
# -----------------------------------------------------------
max_words = 20000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    maxlen=max_len
)

X_test_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=max_len
)

# -----------------------------------------------------------
# 7️⃣ BUILD LSTM MODEL
# -----------------------------------------------------------
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# -----------------------------------------------------------
# 8️⃣ TRAIN MODEL
# -----------------------------------------------------------
history = model.fit(
    X_train_seq, y_train,
    validation_split=0.1,
    epochs=6,
    batch_size=32,
    verbose=1
)

# -----------------------------------------------------------
# 9️⃣ PREDICTIONS
# -----------------------------------------------------------
y_pred = (model.predict(X_test_seq) > 0.5).astype(int).flatten()

# -----------------------------------------------------------
# 🔟 ACCURACY
# -----------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy: {:.2f}%".format(accuracy * 100))

# -----------------------------------------------------------
# 1️⃣1️⃣ CLASSIFICATION REPORT
# -----------------------------------------------------------
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Synthetic"]
))

# -----------------------------------------------------------
# 1️⃣2️⃣ CONFUSION MATRIX
# -----------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Synthetic"],
            yticklabels=["Legitimate", "Synthetic"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(sys.argv[2] + '.png', bbox_inches='tight')
plt.close()

# -----------------------------------------------------------
# 1️⃣3️⃣ TRAINING CURVES (Important for Paper)
# -----------------------------------------------------------
plt.figure(figsize=(12,4))

# Accuracy Curve
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss Curve
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(sys.argv[2] + '.png', bbox_inches='tight')
plt.close()
