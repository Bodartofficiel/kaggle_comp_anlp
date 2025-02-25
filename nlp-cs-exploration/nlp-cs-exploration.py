import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pathlib
from datasets import Dataset
from transformers import AutoTokenizer
from evaluate import load
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

train_dataset_path = pathlib.Path(__name__).parent / "data/train_submission.csv"
df = pd.read_csv(train_dataset_path)
print("Shape :" ,df.shape)
print("Columns :", df.columns)
print("Nombre de langue cible :",len(df["Label"].unique()))
print(df.head())

test_dataset_path = pathlib.Path(__name__).parent / "data/test_without_labels.csv"
test_df = pd.read_csv(test_dataset_path) 
print("Shape : ",test_df.shape)
print(test_df.head())

# Preprocessing : enlever les valeurs manquantes
df.dropna(inplace = True)
print("Shape :", df.shape)
print(df.head())

# Création du mapping label -> id
labels = sorted(df["Label"].unique().tolist())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
df["label"] = df["Label"].map(label2id)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print("Training shape :", train_df.shape)
print("Validation shape :", val_df.shape)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(example):
    return tokenizer(example["Text"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Préparation d'un data collator qui s'occupe du padding dynamique
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(df["label"]), y=df["label"])
class_weights = torch.tensor(class_weights, dtype=torch.float)

class BERTClassifier(nn.Module):
    def __init__(self, model_checkpoint, num_labels):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_checkpoint)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        # Freeze BERT layers (optional for transfer learning)
        for param in self.bert.parameters():
            param.requires_grad = False  # Comment this line if fine-tuning

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(self.dropout(pooled_output))
        return logits

num_labels = len(labels)
model = BERTClassifier(model_checkpoint, num_labels)

accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return accuracy 


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("label")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(outputs.device))  # Apply class weights
        loss = loss_fn(outputs, labels)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# Chargement du fichier test
test_dataset_path = pathlib.Path(__name__).parent / "data/test_without_labels.csv"
test_df = pd.read_csv(test_dataset_path) 
test_df.reset_index(inplace=True)
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Utilisation du modèle pour générer les prédictions
predictions_output = trainer.predict(test_dataset)
preds = np.argmax(predictions_output.predictions, axis=-1)

# Conversion des indices en labels textuels
test_df["Label"] = [id2label[pred] for pred in preds]

# Création du fichier de soumission avec les colonnes "ID" et "Label"
test_df[["index", "Label"]].to_csv("submission.csv", index=False)