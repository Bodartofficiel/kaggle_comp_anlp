from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import json


def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    sentences = []
    labels = []
    for label, texts in data.items():
        sentences.extend(texts)
        labels.extend([label] * len(texts))
    print(sentences[:10])
    print(labels[:10])
    return sentences, labels



# Load classifier
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


def classif_to_label(sequence, candidate_labels):
    result = classifier(sequence, candidate_labels)
    return result['labels'][0]

def classifier_predict_sentences(sentences, sentences_labels):
    y_pred = []
    candidate_labels = list(set(sentences_labels))
    print(len(sentences_labels))
    print(len(sentences))
    classifier_true = 0
    classifier_total = len(sentences)
    for i, sentence in enumerate(sentences):
        y_pred.append(classif_to_label(sentence, candidate_labels))
        if y_pred == sentences_labels[i]:
            classifier_true += 1
    print(f"Classifier accuracy: {classifier_true/classifier_total}")
    return y_pred

def group_by_label(y_pred, sentences_label):
    label_counts = {label: {'true': 0, 'false': 0} for label in set(sentences_label)}
    for pred, true_label in zip(y_pred, sentences_label):
        if pred == true_label:
            label_counts[true_label]['true'] += 1
        else:
            label_counts[true_label]['false'] += 1

    for label, counts in label_counts.items():
        total = counts['true'] + counts['false']
        true_percentage = (counts['true'] / total) * 100 if total > 0 else 0
        false_percentage = (counts['false'] / total) * 100 if total > 0 else 0
        print(f"Label {label}: True = {true_percentage:.2f}%, False = {false_percentage:.2f}%")
        
        
if __name__ == "__main__":

    train_sentences, train_labels = load_data('data/test.json')
    candidate_labels = set(train_labels)
    y_pred = classifier_predict_sentences(train_sentences, train_labels)
    group_by_label(y_pred, train_labels)