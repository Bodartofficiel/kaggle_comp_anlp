import torch
from config import *
from get_and_process_dataset import get_and_process_dataset
from metrics import compute_metrics
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

train_dataset, val_dataset, labels, label2id, id2label = get_and_process_dataset(
    TRAIN_DATA_PATH, tokenizer
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CKPT,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
).to(device)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    eval_strategy="epoch",
    logging_steps=LOGGING_STEPS,
    save_steps=10,
    save_strategy="steps",
    fp16=True,
)

trainer = Trainer(
    model,
    training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

trainer.train()
