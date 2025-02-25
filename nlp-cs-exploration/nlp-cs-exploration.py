import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

df = pd.read_csv("./train_submission.csv")
print("Shape :" ,df.shape)
print("Columns :", df.columns)
print("Nombre de langue cible :",len(df["Label"].unique()))
print(df.head())

df_test = pd.read_csv("./test_without_labels.csv")
print("Shape : ",df_test.shape)
print(df_test.head())

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

num_labels = len(labels)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

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

trainer = Trainer(
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
test_df = pd.read_csv("./test_without_labels.csv") 
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