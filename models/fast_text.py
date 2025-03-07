import fasttext
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
import json
from sklearn.utils import resample
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# Chargement des données
df = pd.read_csv('../data/train_submission.csv')
df.dropna(inplace=True) 
df["Text"] = df["Text"].astype(str)

# Ajouter le format FastText au fichier
df['fasttext_format'] = '__label__' + df['label'] + ' ' + df['Text']
df['fasttext_format_label'] = '__label__' + df['label']


df_train, df_val = train_test_split(df,test_size=0.05,random_state=42)

class_counts = df_train['label'].value_counts()

# Sous-échantillonnage (réduction à 500)
classes_to_downsample = class_counts[class_counts > 500].index
balanced_dfs = []

for label in classes_to_downsample:
    subset = df_train[df_train['label'] == label]
    downsampled = resample(subset, n_samples=500, random_state=42)
    balanced_dfs.append(downsampled)

balanced_dfs.append(df_train[df_train['label'].isin(class_counts[class_counts == 500].index)])
balanced_df_train = pd.concat(balanced_dfs)

# Sauvegarde du fichier sous-échantilloner
balanced_df_train.to_csv("train.csv",index=False)

# Sauvegarder les données dans un fichier temporaire
train_file = f'fasttext_train.txt'
balanced_df_train['fasttext_format'].to_csv(
    train_file, index=False, header=False, sep='\n', quoting=csv.QUOTE_NONE, escapechar='\\'
)  

# Entraîner FastText
model = fasttext.train_supervised(input=train_file, epoch=15, lr=1.5, wordNgrams=2, 
                                    loss='softmax', thread=8, verbose=3, dim=256, 
                                    minCount=100, minCountLabel=0, bucket=5*10**6, 
                                    minn=3, maxn=8)

# Tester FastText
predictions = model.predict(df_val['Text'].tolist(), k=1)[0]

df_val.loc[:, 'pred'] = [pred[0] for pred in predictions]
acc = accuracy_score(df_val['fasttext_format_label'], df_val['pred'])

report = classification_report(df_val['fasttext_format_label'], df_val['pred'], output_dict=True)
report_file = "classification_report.txt"
with open(report_file, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(json.dumps(report))

df_test = pd.read_csv("/../data/test_without_labels.csv")

predictions = model.predict(df_test['Text'].tolist(), k=1)[0]
df_test.loc[:, 'Label'] = [pred[0] for pred in predictions]
df_test['Label'] = df_test['Label'].str.replace("__label__", "", regex=False)
df_test.loc[:, 'ID'] = range(1, len(df_test) + 1)

df_test.to_csv("submission.csv",index=False)

