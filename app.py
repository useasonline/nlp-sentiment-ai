import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Advanced Transformer NLP AI", page_icon="⚡", layout="centered")

st.title("⚡ Advanced Transformer NLP AI")
st.caption("Powered by DistilBERT trained on the public SST-2 dataset.")

# 1. Load the Model using top_k=None (modern standard)
@st.cache_resource
def load_advanced_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        top_k=None
    )

with st.spinner("Initializing neural transformer weights..."):
    nlp_engine = load_advanced_model()

# 2. User Input
st.subheader("Run Inference")
sample_text = st.text_area(
    "Enter complex text (slang, sarcasm, or nuanced sentences):",
    "I expected this to be a complete failure, but against all odds it blew me away."
)

if st.button("Analyze with Deep Learning"):
    if sample_text.strip():
        with st.spinner("Processing self-attention layers..."):
            raw_output = nlp_engine(sample_text)

        # Handle output format safely whether it is nested [[{...}]] or flat [{...}]
        if isinstance(raw_output[0], list):
            predictions = raw_output[0]
        else:
            predictions = raw_output

        # Extract scores into a dictionary
        scores = {item['label'].upper(): float(item['score']) for item in predictions}
        pos_score = scores.get("POSITIVE", 0.0)
        neg_score = scores.get("NEGATIVE", 0.0)

        # Display verdict
        if pos_score > neg_score:
            st.success(f"**Prediction:** POSITIVE (Score: {pos_score * 100:.2f}%)")
        else:
            st.error(f"**Prediction:** NEGATIVE (Score: {neg_score * 100:.2f}%)")

        # Confidence bars
        st.write("#### Neural Confidence Distribution:")
        st.progress(pos_score, text=f"Positive: {pos_score * 100:.1f}%")
        st.progress(neg_score, text=f"Negative: {neg_score * 100:.1f}%")

    else:
        st.warning("Please type a phrase to test.")
