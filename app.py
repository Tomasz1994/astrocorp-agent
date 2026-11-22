import os
import streamlit as st
import requests
import numpy as np
from google import genai
from google.genai import types

# 1. Konfiguracja strony WWW
st.set_page_config(page_title="AstroCorp AI Concierge", page_icon="🚀")
st.title("🚀 AstroCorp AI Concierge")
st.caption("Twój osobisty doradca lotów orbitalnych (RAG + kursy NBP na żywo).")

# 2. Pobieranie klucza API z sejfu Streamlit Secrets
klucz_api = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not klucz_api:
    st.error("Brak klucza GEMINI_API_KEY w ustawieniach Secrets!")
    st.stop()

client = genai.Client(api_key=klucz_api)

# 3. Baza wiedzy RAG (buforowana w pamięci serwera dzięki @st.cache_resource)
@st.cache_resource
def zaladuj_baze_wiedzy():
    dokumenty = [
        "REGULAMIN BAGAŻOWY ASTROCORP: Każdy pasażer lotu na orbitę może zabrać bezpłatnie 1 sztukę bagażu podręcznego o wadze do 4,5 kg. Nadbagaż kosztuje 1200 USD za każdy rozpoczęty kilogram.",
        "POLITYKA ANULOWANIA LOTÓW: Zwrot 100% ceny biletu przysługuje wyłącznie do 30 dni przed startem rakiety. Poniżej 30 dni zwrot wynosi 25%, a na 48 godzin przed startem bilet przepada.",
        "WYMAGANIA ZDROWOTNE: Pasażerowie muszą posiadać zaświadczenie od lekarza medycyny lotniczej wystawione nie wcześniej niż 14 dni przed lotem.",
        "MENU POKŁADOWE: Podczas lotu serwowane są wyłącznie liofilizowane posiłki w tubkach. Zabrania się wnoszenia własnych napojów gazowanych."
    ]
    model_emb = next((m.name for m in client.models.list() if "embedding" in m.name), "text-embedding-004")
    wektory = [
        client.models.embed_content(model=model_emb, contents=dok).embeddings[0].values
        for dok in dokumenty
    ]
    return dokumenty, wektory, model_emb

dokumenty_firmowe, wektory_bazy, model_emb = zaladuj_baze_wiedzy()

# 4. Narzędzia Agenta (RAG + API NBP)
def przeszukaj_baze_wiedzy(zapytanie: str) -> str:
    """Przeszukuje wewnętrzny regulamin firmy AstroCorp w poszukiwaniu zasad lotów, bagażu, zwrotów i menu."""
    w_zap = np.array(client.models.embed_content(model=model_emb, contents=zapytanie).embeddings[0].values)
    wyniki = []
    for i, w_dok in enumerate(wektory_bazy):
        w_dok_np = np.array(w_dok)
        score = np.dot(w_zap, w_dok_np) / (np.linalg.norm(w_zap) * np.linalg.norm(w_dok_np))
        wyniki.append((score, dokumenty_firmowe[i]))
    wyniki.sort(reverse=True, key=lambda x: x[0])
    return "\n".join([tekst for _, tekst in wyniki[:2]])

def pobierz_kurs_nbp(kod_waluty: str) -> float:
    """Pobiera aktualny średni kurs danej waluty (np. 'USD', 'EUR') w PLN z API Narodowego Banku Polskiego."""
    url = f"https://api.nbp.pl/api/exchangerates/rates/a/{kod_waluty.lower()}/?format=json"
    odpowiedz = requests.get(url)
    return odpowiedz.json()['rates'][0]['mid'] if odpowiedz.status_code == 200 else -1.0

# 5. Pamięć sesji użytkownika w przeglądarce (st.session_state)
if "sesja_czatu" not in st.session_state:
    st.session_state.sesja_czatu = client.chats.create(
        model="gemini-flash-lite-latest",
        config=types.GenerateContentConfig(
            system_instruction=(
                "Jesteś przyjaznym i precyzyjnym wirtualnym asystentem linii kosmicznych AstroCorp. "
                "Zasady regulaminu sprawdzaj ZAWSZE w bazie wiedzy, a kursy walut pobieraj z API NBP. "
                "Pamiętaj o regule 'każdego rozpoczętego kilograma' przy obliczaniu nadbagażu."
            ),
            tools=[przeszukaj_baze_wiedzy, pobierz_kurs_nbp],
            temperature=0.1
        )
    )
    st.session_state.historia_ekranu = []

# Wyświetlamy dotychczasowe wiadomości na ekranie
for msg in st.session_state.historia_ekranu:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Pole wpisywania wiadomości na dole ekranu
if pytanie := st.chat_input("Zapytaj o regulamin bagażu, loty lub dopłaty w PLN..."):
    # Pokaż wiadomość użytkownika
    st.session_state.historia_ekranu.append({"role": "user", "content": pytanie})
    with st.chat_message("user"):
        st.markdown(pytanie)

    # Pobierz odpowiedź od Agenta i wyświetl ją
    with st.chat_message("assistant"):
        with st.spinner("Sprawdzam regulamin i kursy NBP..."):
            odpowiedz = st.session_state.sesja_czatu.send_message(pytanie)
            st.markdown(odpowiedz.text)
    st.session_state.historia_ekranu.append({"role": "assistant", "content": odpowiedz.text})
