import glob
import os
import time

import pandas as pd
import torch
from config import *
from get_and_process_dataset import get_and_process_test_dataset
from tqdm import tqdm
from transformers import pipeline

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
test_dataset = get_and_process_test_dataset(TEST_DATA_PATH)

checkpoint_folders = sorted(
    glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
    key=os.path.getmtime,
    reverse=True,
)
latest_checkpoint = checkpoint_folders[0] if checkpoint_folders else None

assert latest_checkpoint, f"No checkpoint found in {OUTPUT_DIR}"

classification_pipeline = pipeline(
    "text-classification", model=latest_checkpoint, device=device
)

batch_size = PREDICTION_BATCH_SIZE
start_time = time.perf_counter()
model_preds = []

with tqdm(total=len(test_dataset["Text"]), desc="Processing") as pbar:
    for i in range(0, len(test_dataset["Text"]), batch_size):
        batch_texts = test_dataset["Text"][i : i + batch_size]
        batch_preds = classification_pipeline(
            batch_texts, truncation=True, max_length=128
        )
        model_preds.extend([s["label"] for s in batch_preds])
        pbar.update(len(batch_texts))

print(f"{time.perf_counter() - start_time:.2f} seconds")

predictions_df = pd.DataFrame(model_preds, columns=["Label"])
predictions_df.to_csv(PRED_OUTPUT_PATH, index=False)
