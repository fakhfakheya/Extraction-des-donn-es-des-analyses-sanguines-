import streamlit as st
import requests
import os

API_URL = "http://127.0.0.1:8000/analyze"  # Ton API FastAPI locale

st.set_page_config(page_title="Analyse OCR Médicale", page_icon="🧪", layout="centered")

# --- Logos sur la même ligne ---
col1, col2 = st.columns([1, 1])
with col1:
    st.image("pixocr_logo.png", width=120)
with col2:
    st.image("logo.png", width=120)

st.title("🧪 OCR Médical + Extraction Automatique")
st.write("Upload un rapport médical scanné (image) pour extraire les valeurs biologiques.")

uploaded_file = st.file_uploader("Choisir une image de rapport", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Image uploadée", use_column_width=True)

    if st.button("Analyser"):
        with st.spinner("Analyse en cours..."):
            # Sauvegarde temporaire
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Envoi à l'API FastAPI
            with open(temp_path, "rb") as f:
                files = {"file": (uploaded_file.name, f, "multipart/form-data")}
                response = requests.post(API_URL, files=files)

            os.remove(temp_path)

            if response.status_code == 200:
                resultats = response.json()
                st.success("✅ Analyse terminée")
                st.json(resultats)
            else:
                st.error(f"Erreur API: {response.status_code}")
