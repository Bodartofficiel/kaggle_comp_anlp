import numpy as np

# Seed
SEED = 1

# Model and tokenizer checkpoint
# MODEL_CKPT = "bert-base-multilingual-cased"
MODEL_CKPT = "xlm-roberta-base"

assert MODEL_CKPT in [
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
], f"Unsupported model name: {MODEL_CKPT}"

# Preprocessing function to use
PREPROCESSING = []
# PREPROCESING = [REGEX, CLEAN_LANGUAGES]

# Column names of train dataframe
COLUMN_NAMES = ["Text", "Label"]

# Training parameters
EPOCHS = 10
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE * 2
PREDICTION_BATCH_SIZE = 512
LOGGING_STEPS = 100
OUTPUT_DIR = MODEL_CKPT + "-finetuned-language-detection"

# Data paths
TRAIN_DATA_PATH = "./data/train_submission.csv"
TEST_DATA_PATH = "./data/test_without_labels.csv"
PRED_OUTPUT_PATH = "./data/submission_val.csv"
