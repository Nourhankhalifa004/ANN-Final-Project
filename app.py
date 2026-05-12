import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.backend import clear_session
import pickle, os, gc
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_DIR   = os.path.join(BASE_DIR, 'saved_models')
PROCESSED   = os.path.join(BASE_DIR, 'data', 'processed')

st.set_page_config(page_title="Smart Game Review Analyzer", layout="wide")
st.title("🎮 Smart Game Review Analyzer")

@st.cache_resource
def load_models():
    m1 = load_model(os.path.join(SAVED_DIR, 'M1_extractor.keras'), compile=False)
    m2 = load_model(os.path.join(SAVED_DIR, 'M2_extractor.keras'), compile=False)
    m3 = load_model(os.path.join(SAVED_DIR, 'M3_extractor.keras'), compile=False)
    m4 = load_model(os.path.join(SAVED_DIR, 'M4_extractor.keras'), compile=False)
    m5 = load_model(os.path.join(SAVED_DIR, 'M5_final.keras'),     compile=False)

    m1_best = load_model(os.path.join(SAVED_DIR, 'M1_best.keras'),  compile=False)
    m2_full = load_model(os.path.join(SAVED_DIR, 'M2_best.keras'),  compile=False)
    m3_full = load_model(os.path.join(SAVED_DIR, 'M3_best.keras'),  compile=False)

    with open(os.path.join(PROCESSED, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    return m1, m2, m3, m4, m5, m1_best, m2_full, m3_full, tokenizer

m1_ext, m2_ext, m3_ext, m4_ext, m5, m1_best, m2_full, m3_full, tokenizer = load_models()

# ── UI ────────────────────────────────────────────────────────────────────────
col_img, col_text = st.columns(2)

with col_img:
    st.subheader("🖼 Game Cover Image")
    uploaded_img = st.file_uploader("Upload game header image", type=["jpg","jpeg","png"])

with col_text:
    st.subheader("📝 Review Text")
    review_text = st.text_area("Paste the review here", height=160)

st.subheader("📊 Tabular Features")
c1, c2, c3, c4 = st.columns(4)
playtime    = c1.number_input("Playtime (hours)",         value=10.0,  min_value=0.0)
rev_score   = c2.number_input("Review Score (0–10)",      value=7.0,   min_value=0.0, max_value=10.0)
price       = c3.number_input("Game Price ($)",           value=29.99, min_value=0.0)
helpful_votes = c4.number_input("Helpful Votes",          value=5,     min_value=0)

run = st.button("🔍 Analyse Review", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
IMG_SIZE = 224
MAX_LEN  = 200

def preprocess_image(file):
    img = Image.open(file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]         

def preprocess_text(text, max_len=MAX_LEN):
    from keras.utils import pad_sequences
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded.astype(np.int32)       

def build_features(review_text, playtime, rev_score, price, helpful_votes):

    words      = review_text.split()
    word_count = len(words)
    feats = np.array([
        rev_score / 10.0,           # 0 review_score_norm
        np.log1p(playtime),         # 1 playtime_norm
        float(word_count),          # 2 word_count
        1.0 if word_count < 5 else 0.0,  # 3 is_repetitive
        price / 100.0,              # 4 price_norm
        np.log1p(helpful_votes),    # 5 review_count_log
        0.75,                       # 6 positive_ratio (neutral)
        1.0,                        # 7 has_achievements (neutral)
        np.log1p(playtime),         # 8 median_playtime_norm
        0.0,                        # 9 required_age_norm (neutral)
    ], dtype=np.float32).reshape(1, 10)
    return feats

if run:
    if not uploaded_img:
        st.warning("Please upload a game cover image.")
        st.stop()
    if not review_text.strip():
        st.warning("Please enter a review.")
        st.stop()

    with st.spinner("Running models…"):

        # ── Pre-process inputs ────────────────────────────────────────────────
        img_arr  = preprocess_image(uploaded_img)
        text_arr = preprocess_text(review_text)
        feat_arr = build_features(review_text, playtime, rev_score, price, helpful_votes)

        # ── M1: Image Match ───────────────────────────────────────────────────
        feat_img         = m1_ext.predict(img_arr, verbose=0)        # extractor embedding → M5
        match_proba_full = m1_best.predict(img_arr, verbose=0)        # classifier → display
        image_match_pct  = float(match_proba_full[0][1]) * 100        # class 1 = Match

        # ── M2: Sentiment ─────────────────────────────────────────────────────
        feat_text      = m2_ext.predict(text_arr, verbose=0)    
        sent_proba     = m2_full.predict(text_arr, verbose=0)   
        # FIX: index 1 = Positive, index 0 = Negative
        sentiment_pos  = float(sent_proba[0][1]) * 100
        sentiment_lbl  = "Positive" if sent_proba[0][1] > 0.5 else "Negative"

        # ── M3: Fake Detection ────────────────────────────────────────────────
        feat_gru       = m3_ext.predict(text_arr, verbose=0)     
        fake_proba     = m3_full.predict(text_arr, verbose=0)    
        # FIX: index 0 = Real/Authentic, index 1 = Fake
        real_pct       = float(fake_proba[0][0]) * 100
        auth_lbl       = "Authentic" if fake_proba[0][0] > 0.5 else "Fake"

        # ── M4: Tabular features ──────────────────────────────────────────────
        feat_ann       = m4_ext.predict(feat_arr, verbose=0)    

        # ── M5: Fusion ────────────────────────────────────────────────────────
        trust_proba = m5.predict(
            [feat_img, feat_text, feat_gru, feat_ann], verbose=0
        )

        trust_score = float(trust_proba[0][2]) * 100
        trust_class = int(np.argmax(trust_proba[0]))
        trust_labels = {0: "🟥 FAKE", 1: "🟨 SUSPICIOUS", 2: "🟩 TRUSTED"}
        trust_lbl   = trust_labels[trust_class]

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Image Match",       f"{image_match_pct:.1f}%")
    r2.metric("Sentiment",         sentiment_lbl,  f"↑ {sentiment_pos:.1f}% Pos")
    r3.metric("Authenticity",      auth_lbl,       f"↑ {real_pct:.1f}% Real")
    r4.metric("Final Trust Score", f"{trust_score:.1f}%", trust_lbl)

    # Confidence breakdown
    st.markdown("#### M5 Fusion Confidence Breakdown")
    conf_cols = st.columns(3)
    conf_cols[0].progress(int(trust_proba[0][0]*100), text=f"Fake: {trust_proba[0][0]*100:.1f}%")
    conf_cols[1].progress(int(trust_proba[0][1]*100), text=f"Suspicious: {trust_proba[0][1]*100:.1f}%")
    conf_cols[2].progress(int(trust_proba[0][2]*100), text=f"Trusted: {trust_proba[0][2]*100:.1f}%")
