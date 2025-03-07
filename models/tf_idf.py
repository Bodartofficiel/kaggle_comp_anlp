import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import time

# Charger uniquement les colonnes nécessaires
df = pd.read_csv('data/train_submission.csv', usecols=['Text', 'Label'])

# Supprimer les lignes où il n'y a qu'une seule occurrence de chaque label
label_counts = df['Label'].value_counts()
df = df[df['Label'].isin(label_counts[label_counts > 1].index)]

# Prendre la moitié des données, stratifiées par label

# Prétraitement : conversion en minuscules et suppression des NaN
df['Text'] = df['Text'].astype(str).str.lower()
df = df.dropna(subset=['Label'])

# Conversion de la colonne Label en type catégorie pour économiser de la mémoire
df['Label'] = df['Label'].astype('category')

# Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=0.15, random_state=123)

# Optimisation de TF-IDF
tfidf = TfidfVectorizer(
    max_features=500,  # Augmenter légèrement pour plus d'information
    sublinear_tf=True,  # Rend les fréquences TF plus stables
)

# Optimisation du RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=50,       # Réduire légèrement le nombre d'arbres
    max_depth=30,          # Limiter la profondeur pour éviter trop de RAM
    n_jobs=-1,             # Utiliser tous les cœurs disponibles
    random_state=123
)

# Création du pipeline
pipeline = make_pipeline(tfidf, rf)

# Entraînement du modèle
print('Fitting...')
start_time = time.time()
pipeline.fit(X_train, y_train)
fit_time = time.time() - start_time
print(f"Fit time: {fit_time:.4f} seconds")

# Prédictions sur le test
print('Predicting...')
start_time = time.time()
y_pred = pipeline.predict(X_test)
predict_time = time.time() - start_time
print(f"Predict time: {predict_time:.4f} seconds")

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
