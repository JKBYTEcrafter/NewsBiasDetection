import sys
"""Gemini"""

!pip install -U google-generativeai

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
df = pd.read_excel(sys.argv[1], header=None)

texts = df.iloc[:, 0].astype(str).tolist()
original_labels = df.iloc[:, 2].astype(str).tolist()

# Combine labels
binary_labels = [
    0 if label == "Legitimate" else 1
    for label in original_labels
]

data = pd.DataFrame({
    "text": texts,
    "label": binary_labels
})

print(data["label"].value_counts())
train_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
training_args = TrainingArguments(
    output_dir="./muril_bias_model",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="no",
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

trainer.train()
preds = trainer.predict(test_ds)

y_pred = np.argmax(preds.predictions, axis=1)
y_true = preds.label_ids

accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {accuracy:.4f}\n")

print("Classification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Pure", "Septic"]
))
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pure", "Septic"],
    yticklabels=["Pure", "Septic"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MuRIL Confusion Matrix")
plt.savefig(sys.argv[2] + '.png', bbox_inches='tight')
plt.close()
