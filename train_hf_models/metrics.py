from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    """Custom metric to be used during training."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}
