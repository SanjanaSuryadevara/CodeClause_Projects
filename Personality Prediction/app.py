import os, re, io, json, joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import docx
except Exception:
    docx = None

TRAITS = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]

TFIDF_PATH = "tfidf_b5.pkl"
MODEL_PATH = "bigfive_ridge.pkl"
LABEL_SCALER_PATH = "label_minmax.pkl"  # not required for inference but kept for completeness

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load(TFIDF_PATH)
    model = joblib.load(MODEL_PATH)
    return tfidf, model

def clean_text(x: str) -> str:
    if not isinstance(x, str): return ""
    x = re.sub(r'http\S+', ' ', x)
    x = re.sub(r'[\r\n\t]+', ' ', x)
    x = re.sub(r'[^a-zA-Z ]+', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip().lower()
    return x

def read_pdf_bytes(data: bytes) -> str:
    if fitz is None: return data.decode(errors="ignore")
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n".join(p.get_text("text") for p in doc)

def read_docx_bytes(data: bytes) -> str:
    if docx is None: return data.decode(errors="ignore")
    d = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in d.paragraphs)

def vector_predict(text: str, tfidf, model):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    preds = np.clip(model.predict(vec)[0], 0, 1)
    return {t: float(preds[i]) for i,t in enumerate(TRAITS)}

def radar_chart(scores: dict):
    r = [scores[t] for t in TRAITS]
    fig = go.Figure(data=go.Scatterpolar(r=r+[r[0]], theta=TRAITS+[TRAITS[0]], fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                      showlegend=False, margin=dict(l=20,r=20,t=20,b=20), height=420)
    return fig

st.set_page_config(page_title="Personality from CV", page_icon="🧠", layout="wide")
st.title("🧠 Personality Prediction from CV")
st.caption("For research/education only — not for hiring decisions.")

# Load artifacts (error if not present)
try:
    tfidf, model = load_artifacts()
except Exception as e:
    st.error("Artifacts not found. Make sure 'tfidf_b5.pkl' and 'bigfive_ridge.pkl' are in the same folder as app.py.")
    st.stop()

left, right = st.columns([0.55, 0.45])

with left:
    st.subheader("Upload or Paste Resume Text")
    up = st.file_uploader("Upload (TXT/PDF/DOCX)", type=["txt","pdf","docx"])
    txt = st.text_area("Or paste text here", height=240)
    run = st.button("Analyze", type="primary")

if run:
    if up is not None:
        data = up.read()
        if up.name.lower().endswith(".pdf"): text = read_pdf_bytes(data)
        elif up.name.lower().endswith(".docx"): text = read_docx_bytes(data)
        else: text = data.decode(errors="ignore")
    else:
        text = txt
    if not text.strip():
        st.warning("Please upload a file or paste text.")
    else:
        scores = vector_predict(text, tfidf, model)
        with right:
            st.subheader("Results")
            st.plotly_chart(radar_chart(scores), use_container_width=True)
            for t,v in scores.items():
                band = "higher tendency" if v>=0.65 else ("balanced" if v>=0.45 else "lower tendency")
                st.markdown(f"**{t}**: {v:.2f} — {band}")
