# =========================================================
# File : train_sms_spam.py
# Deskripsi : Training model klasifikasi SMS spam Bahasa Indonesia
# Output : model_sms.pkl
# =========================================================

import csv
import re
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ---------------------------------------------------------
# 1. Persiapan Sastrawi
# ---------------------------------------------------------
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()
factory_stop = StopWordRemoverFactory()
stopwords = set(factory_stop.get_stop_words())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [stemmer.stem(word) for word in text.split() if word not in stopwords]
    return ' '.join(tokens)

# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------
namaFile = "dataset_sms_spam_v1.csv"  # ubah sesuai lokasi file
data, label = [], []

with open(namaFile, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        data.append(clean_text(row[0]))
        label.append(row[1])

print("Jumlah data:", len(data))
print(Counter(label))

# ---------------------------------------------------------
# 3. Split data
# ---------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. Buat pipeline model
# ---------------------------------------------------------
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB())
])

# ---------------------------------------------------------
# 5. Training
# ---------------------------------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print("\n=== HASIL TRAINING ===")
print("Akurasi :", round(akurasi, 4))
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 6. Simpan model
# ---------------------------------------------------------
joblib.dump(model, "model_sms.pkl")
print("\n✅ Model tersimpan ke file: model_sms.pkl")
