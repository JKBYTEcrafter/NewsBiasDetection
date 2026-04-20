import sys
"""Binary CNN"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_excel(sys.argv[1])  # Replace with actual path

# 2. Extract Hindi sentences and original labels
texts = df.iloc[:, 0].astype(str).tolist()
original_labels = df.iloc[:, 2].astype(str).tolist()

# 3. Merge Partially Legitimate and Synthetic into one class: 'Synthetic'
binary_labels = ['Legitimate' if label == 'Legitimate' else 'Synthetic' for label in original_labels]

# 4. Label encode the binary classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(binary_labels)  # Legitimate=0, Synthetic=1
y_categorical = to_categorical(y_encoded, num_classes=2)

# 5. Tokenize and pad text
vocab_size = 10000
max_length = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# 7. Build CNN model
embedding_dim = 100
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Binary output
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 8. Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# 9. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy (Legitimate vs Synthetic):", accuracy)

# 10. Optional: Classification report
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Decode to label names
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))
