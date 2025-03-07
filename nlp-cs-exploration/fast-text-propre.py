
import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import re
import emoji
import os

def balance_dataset_downsample(df, n_samples=500, random_state=42):
    df_balanced = pd.concat([
        group if len(group) < n_samples else resample(group, replace=False, n_samples=n_samples, random_state=random_state)
        for _, group in df.groupby("Label")
    ])
    return df_balanced

def preprocess_text(text):
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('"', ' ')
    text = emoji.replace_emoji(text, "")
    text = re.sub(r'[@#]\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    return text

def prepare_data(file_path, min_class_size=500):
    df = pd.read_csv(file_path)
    df['Text'] = df['Text'].apply(preprocess_text)
    
    class_counts = df["Label"].value_counts()
    classes_to_downsample = class_counts[class_counts > min_class_size].index
    df_to_downsample = df[df["Label"].isin(classes_to_downsample)]
    df = pd.concat([df_to_downsample, df[~df["Label"].isin(classes_to_downsample)]])
    
    df['fasttext_format'] = '__label__' + df['Label'] + ' ' + df['Text']
    return df

def train_fasttext_model(input_file, output_model,retrain=True, epoch=9, lr=2, wordNgrams=1, loss='softmax', thread=10, dim=256, minCount=1000, minCountLabel=0, bucket=10**6, minn=1, maxn=5):
    if os.path.exists(output_model) and retrain == False:
        model = fasttext.load_model(output_model)
    else:
        model = fasttext.train_supervised(input=input_file, epoch=epoch, lr=lr, wordNgrams=wordNgrams, loss=loss, thread=thread, dim=dim, minCount=minCount, minCountLabel=minCountLabel, bucket=bucket, minn=minn, maxn=maxn)
        model.save_model(output_model)
    return model

def evaluate_model(model, df_val):
    text = df_val['Text'].tolist()
    predictions = model.predict(text, k=1)
    df_val['prediction'] = predictions[0]
    df_val['prediction_prob'] = predictions[1]
    df_val['prediction'] = df_val['prediction'].apply(lambda x: x[0])
    
    # df_val['fasttext_format_label'] = df_val['fasttext_format_label'].astype(str)
    df_val['Label'] = df_val['Label'].astype(str)
    df_val['prediction'] = df_val['prediction'].apply(lambda x: x[9:12]).astype(str)
    
    accuracy = accuracy_score(df_val['Label'], df_val['prediction'])
    classif_report = classification_report(df_val['Label'], df_val['prediction'])
    return accuracy, classif_report

def mkdir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def main(data_folder, train_file):
    train_file_path = os.path.join(data_folder, train_file)
    df = prepare_data(train_file_path)
    
    df_train, df_val = train_test_split(df, test_size=0.15, random_state=42)
    # df_train = balance_dataset_downsample(df_train, n_samples=500, random_state=42)
    
    dump_path = os.path.join(data_folder, 'dump_fasttext.txt')
    df_train['fasttext_format'].to_csv(dump_path, index=False, header=False, sep='\n')
    
    print(f"✅ Fichier '{dump_path}' généré avec succès !")
    
    models_dir = os.path.join(data_folder, 'models_dump')
    mkdir_if_not_exists(models_dir)
    model_filename = 'fine_tuned_model.bin'
    model = train_fasttext_model(input_file=dump_path, output_model=os.path.join(models_dir,model_filename), retrain=True)
    
    accuracy, classif_report = evaluate_model(model, df_val)
    # print(classif_report)
    print(accuracy)

if __name__ == "__main__":
    data_folder = "data"
    train_file = "train_submission.csv"
    
    main(data_folder, train_file)
