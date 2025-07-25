import streamlit as st
import pandas as pd
from transformers import pipeline

# 🔄 Ładowanie modeli
@st.cache_resource
def load_stylizer():
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="Falconsai/text_summarization")

stylizer = load_stylizer()
summarizer = load_summarizer()

# 📂 Wczytaj dane
df = pd.read_csv("/Users/julia/Documents/spotify_millsongdata.csv")

if "text" not in df.columns:
    st.error("Brak kolumny 'text' – dataset nie zawiera tekstów piosenek.")
    st.stop()

df = df.dropna(subset=["text"])
available_songs = df["song"].dropna().unique().tolist()

# 🎭 Style emocjonalne
styles = ["Smutny", "Wesoły", "Romantyczny", "Mroczny", "Poetycki"]

# 🧠 Rodzaje przekształceń
modes = ["Zmień styl/emocje", "Streszcz piosenkę"]

# 🎵 Interfejs aplikacji
st.title("🎶 Przekształcanie tekstów piosenek")

# 🔘 Wybór operacji
selected_mode = st.radio("Co chcesz zrobić?", modes)

# 🎶 Wybór piosenki
selected_song = st.selectbox("Wybierz piosenkę:", available_songs)
selected_lyrics = df[df["song"] == selected_song]["text"].values[0]

# 🎯 Stylizacja
if selected_mode == "Zmień styl/emocje":
    selected_style = st.selectbox("Wybierz styl/emocje:", styles)
    if st.button("Przekształć tekst"):
        prompt = f"Przekształć poniższy tekst piosenki w stylu {selected_style}:\n\n{selected_lyrics}"
        result = stylizer(prompt, max_new_tokens=300)[0]["generated_text"]
        st.success(f"✅ Piosenka „{selected_song}” została przekształcona na styl: **{selected_style}**")
        st.text_area("✏️ Nowy tekst piosenki:", result, height=400)

# ✨ Streszczenie
elif selected_mode == "Streszcz piosenkę":
    if st.button("Stwórz streszczenie"):
        result = summarizer(selected_lyrics, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        st.success(f"✅ Oto streszczenie piosenki „{selected_song}”")
        st.text_area("📝 Streszczenie:", result, height=200)
