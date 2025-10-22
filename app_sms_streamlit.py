# =========================================================
# File : app_sms_streamlit.py
# Deskripsi : Aplikasi Streamlit untuk klasifikasi SMS spam
# Cara menjalankan:
#    streamlit run app_sms_streamlit.py
# =========================================================

import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ---------------------------------------------------------
# 1. Inisialisasi Stemmer & Stopwords
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
# 2. Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model_sms.pkl")

model = load_model()

# ---------------------------------------------------------
# 3. UI Streamlit
# ---------------------------------------------------------
st.set_page_config(page_title="Deteksi SMS Spam", page_icon="📱")
st.title("📱 Deteksi SMS Spam Bahasa Indonesia")
st.markdown("Masukkan teks SMS untuk memeriksa apakah termasuk **normal**, **penipuan**, atau **promosi**.")

# Input teks
sms_input = st.text_area("Ketik isi SMS di sini...", height=150)

if st.button("🔍 Prediksi"):
    if sms_input.strip() == "":
        st.warning("Silakan masukkan teks SMS terlebih dahulu.")
    else:
        cleaned = clean_text(sms_input)
        pred = model.predict([cleaned])[0]

        # Tampilan hasil
        if pred == "penipuan":
            st.error("🚨 SMS ini terdeteksi sebagai **PENIPUAN**.")
        elif pred == "promosi":
            st.warning("📢 SMS ini kemungkinan **PROMOSI / IKLAN**.")
        else:
            st.success("✅ SMS ini terdeteksi **NORMAL / AMAN**.")

        st.markdown("---")
        st.markdown("🧠 *Model menggunakan TF-IDF + Multinomial Naive Bayes (Scikit-Learn)*")

