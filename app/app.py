import streamlit as st
from sentence_transformers import SentenceTransformer, util
from bert_score import score
import torch
import numpy as np
import time

# ===================================================
# PROMETHEUS MONITORING
# ===================================================
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
REQUEST_COUNT = Counter(
    "similarity_requests_total",
    "Total number of similarity computations"
)

REQUEST_LATENCY = Histogram(
    "similarity_latency_seconds",
    "Time spent computing similarity"
)

# Start Prometheus metrics server at :8000/metrics
start_http_server(8000)


# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="NSU AI • Paraphrase Similarity",
    page_icon="🤖",
    layout="wide"
)

# ===================================================
# CUSTOM CSS – Neon + Glassmorphism + Dark Theme
# ===================================================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0e0e0e, #1b1b1b, #111) !important;
}

.main-title {
    font-size: 52px;
    font-weight: 900;
    text-align: center;
    color: #00eaff;
    text-shadow: 0px 0px 20px #00eaff;
    margin-bottom: -5px;
}

.subtitle {
    text-align: center;
    color: #cfcfcf;
    font-size: 20px;
    margin-bottom: 40px;
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0px 4px 30px rgba(0, 255, 200, 0.1);
}

.similarity-score {
    font-size: 40px;
    font-weight: 800;
    color: #00ffaa;
    text-align: center;
}

.footer {
    text-align: center;
    color: #999;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ===================================================
# MODEL PATHS
# ===================================================
MODEL_PATHS = {
    "LaBSE (Self-hosted)": "/app/models/labse_ahsan_baseline",
}


# ===================================================
# LOAD MODEL
# ===================================================
@st.cache_resource
def load_model(path):
    with st.spinner("⚡ Loading selected model... please wait..."):
        model = SentenceTransformer(path)
    return model


# ===================================================
# UI HEADER
# ===================================================
st.markdown("<h1 class='main-title'>NSU AI • Semantic Similarity App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by your <b>self-hosted</b> LaBSE model • Modern UI</p>", unsafe_allow_html=True)


# ===================================================
# SIDEBAR – Settings
# ===================================================
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Choose Model",
        list(MODEL_PATHS.keys())
    )

    enable_bertscore = st.checkbox("Compute BERTScore F1", value=True)

    st.info(" You are using a **self-hosted** model. No HuggingFace download required.")


model = load_model(MODEL_PATHS[model_choice])


# ===================================================
# INPUT PANEL
# ===================================================
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### Enter Text for Similarity Comparison")

col1, col2 = st.columns(2)
with col1:
    sentence1 = st.text_area("🔹 Sentence 1", height=150)
with col2:
    sentence2 = st.text_area("🔸 Sentence 2", height=150)

st.markdown("</div>", unsafe_allow_html=True)


# ===================================================
# COMPUTE SIMILARITY
# ===================================================
if st.button("Compute Similarity", use_container_width=True):

    if not sentence1 or not sentence2:
        st.error("Please enter both sentences.")
    else:
        
        REQUEST_COUNT.inc()
        start_time = time.time()

        with st.spinner("🔍 Computing embeddings..."):
            emb1 = model.encode(sentence1, convert_to_tensor=True)
            emb2 = model.encode(sentence2, convert_to_tensor=True)
            cos_sim = util.cos_sim(emb1, emb2)[0][0].item()

        # PROMETHEUS: latency
        REQUEST_LATENCY.observe(time.time() - start_time)

        # -------------------------
        # COSINE SIMILARITY
        # -------------------------
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Cosine Similarity Score")
        st.markdown(f"<p class='similarity-score'>{cos_sim:.4f}</p>", unsafe_allow_html=True)

        if cos_sim > 0.80:
            st.success("Sentences are very similar!")
        elif cos_sim > 0.50:
            st.info("Sentences are somewhat similar.")
        else:
            st.warning("⚡ Sentences are not very similar.")

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # OPTIONAL BERTScore F1
        # -------------------------
        if enable_bertscore:
            with st.spinner("Computing BERTScore..."):
                P, R, F1 = score([sentence1], [sentence2], lang="en", verbose=False)
                f1_value = F1.mean().item()

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### BERTScore F1")
            st.markdown(f"<p class='similarity-score'>{f1_value:.4f}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ===================================================
# FOOTER
# ===================================================
st.markdown("<p class='footer'>Made by Ahsan Shakoor • NSU AI • 2025</p>", unsafe_allow_html=True)
