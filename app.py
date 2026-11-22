import os
import time
import streamlit as st
import requests
import numpy as np
from google import genai
from google.genai import types

# 1. Konfiguracja strony WWW
st.set_page_config(page_title="AstroCorp AI Concierge", page_icon="🚀")
st.title("🚀 AstroCorp AI Concierge")
st.caption("Twój osobisty doradca lotów orbitalnych (RAG + kursy NBP na żywo).")

# Lista modeli w kolejności priorytetu (Fallback w razie przeciążenia 503)
DOSTEPNE_MODELE = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash"
]

# 2. Pobieranie klucza API z sejfu Streamlit Secrets
klucz_api = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not klucz_api:
    st.error("Brak klucza GEMINI_API_KEY w ustawieniach Secrets!")
    st.stop()

if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=klucz_api)

client = st.session_state.client

# 3. Baza wiedzy RAG (buforowana w pamięci serwera)
@st.cache_resource
def zaladuj_baze_wiedzy(_klient):
    dokumenty = [
        "REGULAMIN BAGAŻOWY ASTROCORP: Każdy pasażer lotu na orbitę może zabrać bezpłatnie 1 sztukę bagażu podręcznego o wadze do 4,5 kg. Nadbagaż kosztuje 1200 USD za każdy rozpoczęty kilogram.",
        "POLITYKA ANULOWANIA LOTÓW: Zwrot 100% ceny biletu przysługuje wyłącznie do 30 dni przed startem rakiety. Poniżej 30 dni zwrot wynosi 25%, a na 48 godzin przed startem bilet przepada.",
        "WYMAGANIA ZDROWOTNE: Pasażerowie muszą posiadać zaświadczenie od lekarza medycyny lotniczej wystawione nie wcześniej niż 14 dni przed lotem.",
        "MENU POKŁADOWE: Podczas lotu serwowane są wyłącznie liofilizowane posiłki w tubkach. Zabrania się wnoszenia własnych napojów gazowanych."
    ]
    model_emb = next((m.name for m in _klient.models.list() if "embedding" in m.name), "text-embedding-004")
    wektory = [
        _klient.models.embed_content(model=model_emb, contents=dok).embeddings[0].values
        for dok in dokumenty
    ]
    return dokumenty, wektory, model_emb

dokumenty_firmowe, wektory_bazy, model_emb = zaladuj_baze_wiedzy(client)

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

# 5. Pamięć rozmowy (lista wiadomości przechowywana niezależnie od wybranego modelu!)
if "historia_rozmowy" not in st.session_state:
    st.session_state.historia_rozmowy = []

for msg in st.session_state.historia_rozmowy:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Funkcja wysyłająca zapytanie z automatycznym przełączaniem modeli (Fallback)
def zapytaj_agenta_z_fallbackiem(historia):
    konfiguracja = types.GenerateContentConfig(
        system_instruction=(
            "Jesteś przyjaznym i precyzyjnym wirtualnym asystentem linii kosmicznych AstroCorp. "
            "Zasady regulaminu sprawdzaj ZAWSZE w bazie wiedzy, a kursy walut pobieraj z API NBP. "
            "Pamiętaj o regule 'każdego rozpoczętego kilograma' przy obliczaniu nadbagażu."
        ),
        tools=[przeszukaj_baze_wiedzy, pobierz_kurs_nbp],
        temperature=0.1
    )

    # Budujemy pełną historię dla API (zamieniamy 'assistant' na 'model' zgodnie z wymogiem Gemini)
    zawartosc_api = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=m["content"])]
        )
        for m in historia
    ]

    ostatni_blad = None
    for model_name in DOSTEPNE_MODELE:
        try:
            odpowiedz = client.models.generate_content(
                model=model_name,
                contents=zawartosc_api,
                config=konfiguracja
            )
            return odpowiedz.text, model_name
        except Exception as e:
            ostatni_blad = e
            # Jeśli serwer jest przeciążony (503) lub limit chwilowo przekroczony (429), próbujemy kolejny model
            if "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e):
                time.sleep(0.5)
                continue
            else:
                raise e  # Inny błąd (np. literówka w kodzie) rzucamy od razu
    raise ostatni_blad

# 7. Obsługa okna czatu
if pytanie := st.chat_input("Zapytaj o regulamin bagażu, loty lub dopłaty w PLN..."):
    st.session_state.historia_rozmowy.append({"role": "user", "content": pytanie})
    with st.chat_message("user"):
        st.markdown(pytanie)

    with st.chat_message("assistant"):
        with st.spinner("Sprawdzam regulamin i kursy NBP..."):
            try:
                tekst_odpowiedzi, uzyty_model = zapytaj_agenta_z_fallbackiem(st.session_state.historia_rozmowy)
                st.markdown(tekst_odpowiedzi)
                st.caption(f"⚙️ Obsłużono przez: `{uzyty_model}`")
                st.session_state.historia_rozmowy.append({"role": "assistant", "content": tekst_odpowiedzi})
            except Exception as e:
                st.error(f"Wszystkie modele są chwilowo zajęte. Szczegóły: {e}")
