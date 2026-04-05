import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model('model.h5')

MAX_TEXT_LEN = 100

# Session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Clear callback
def clear_all():
    st.session_state.prediction = None
    st.session_state.input_text = ""

st.title("📩 Spam vs Ham Detector")

# Input box
user_input = st.text_area("Enter your message:", key="input_text")

# ✅ Side-by-side buttons
col_space1, col1, col2, col_space2 = st.columns([2, 1, 1, 2])


with col1:
    predict_clicked = st.button("🔍 Predict")

with col2:
    st.button("🧹 Clear", on_click=clear_all)

# Handle prediction
if predict_clicked:
    # Auto-clear previous prediction
    st.session_state.prediction = None

    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_TEXT_LEN)

        pred = model.predict(padded)[0][0]
        st.session_state.prediction = pred

# Show result
if st.session_state.prediction is not None:
    pred = st.session_state.prediction

    if pred > 0.5:
        st.error(f"🚨 Spam (Confidence: {pred:.2f})")
    else:
        st.success(f"✅ Ham (Confidence: {1 - pred:.2f})")