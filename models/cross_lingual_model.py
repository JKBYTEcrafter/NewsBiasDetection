import sys
"""Cross Lingual"""

pip install pandas scikit-learn torch transformers datasets openpyxl matplotlib

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel(sys.argv[1])

texts = df.iloc[:, 0].astype(str)
labels_raw = df.iloc[:, 2].astype(str)

labels = labels_raw.apply(
    lambda x: "Legitimate" if x.strip() == "Legitimate" else "Synthetic"
)

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

print("Label mapping:",
      dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, stratify=y, random_state=42
)

train_df = pd.DataFrame({"text": X_train, "label": y_train})
test_df = pd.DataFrame({"text": X_test, "label": y_test})

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# -----------------------------
# TOKENIZER
# -----------------------------
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch",
                    columns=["input_ids","attention_mask","label"])
test_ds.set_format(type="torch",
                   columns=["input_ids","attention_mask","label"])

# -----------------------------
# MODEL
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# -----------------------------
# TRAINING ARGS (SAFE VERSION)
# -----------------------------
training_args = TrainingArguments(
    output_dir="./bias_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

# -----------------------------
# TRAIN
# -----------------------------
trainer.train()

# -----------------------------
# PREDICT
# -----------------------------
pred = trainer.predict(test_ds)

y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=1)

# -----------------------------
# METRICS
# -----------------------------
print("\nAccuracy:", accuracy_score(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n",
      classification_report(y_true, y_pred,
                            target_names=encoder.classes_))

# -----------------------------
# PLOT MATRIX
# -----------------------------
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.savefig(sys.argv[2] + '.png', bbox_inches='tight')
plt.close()

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_sentence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return encoder.inverse_transform([pred])[0]

print(predict_sentence("यह खबर पूरी तरह सही है"))