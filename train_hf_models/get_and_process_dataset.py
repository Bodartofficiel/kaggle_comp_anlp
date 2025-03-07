import pandas as pd
from config import COLUMN_NAMES, PREPROCESSING, SEED
from datasets import Dataset
from transformers import AutoTokenizer

from data_processing.cleaning import clean_languages, clean_sentences


def get_and_process_dataset(file_path: str, tokenizer: AutoTokenizer):
    dataset = (
        pd.read_csv(file_path, encoding="utf-8").dropna().reset_index()[COLUMN_NAMES]
    )
    labels = dataset[COLUMN_NAMES[1]].unique()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    dataset = Dataset.from_pandas(dataset)

    dataset = dataset.map(
        lambda x: preprocess_dataset(
            x, PREPROCESSING, tokenizer, COLUMN_NAMES, label2id
        ),
        batched=True,
        remove_columns=COLUMN_NAMES,
    )
    dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
    return dataset["train"], dataset["test"], labels, label2id, id2label


def get_and_process_test_dataset(file_path: str):
    dataset = Dataset.from_csv(file_path).remove_columns(["Usage"])
    preprocessing_func = lambda x: x
    if "REGEX" in PREPROCESSING:
        preprocessing_func = lambda x: clean_sentences(preprocessing_func(x))
    if "CLEAN_LANGUAGES" in PREPROCESSING:
        preprocessing_func = lambda x: clean_languages(preprocessing_func(x))
    return dataset.map(preprocessing_func, batched=False)


def preprocess_dataset(
    batch,
    preprocessing: list,
    tokenizer: AutoTokenizer,
    column_names: list,
    label2id: dict,
):
    text_col, label_col = column_names
    preprocessing_func = lambda x: x
    if "REGEX" in preprocessing:
        preprocessing_func = lambda x: clean_sentences(preprocessing_func(x))
    if "CLEAN_LANGUAGES" in preprocessing:
        preprocessing_func = lambda x: clean_languages(preprocessing_func(x))
    return {
        "input_ids": tokenizer(
            list(
                map(preprocessing_func, batch[text_col]),
            ),
            truncation=True,
            max_length=128,
        )["input_ids"],
        "labels": list(map(lambda x: label2id[x], batch[label_col])),
    }


if __name__ == "__main__":
    from config import MODEL_CKPT, TRAIN_DATA_PATH

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    print(get_and_process_dataset(TRAIN_DATA_PATH, tokenizer))
