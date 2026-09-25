import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Advanced Transformer NLP AI", page_icon="⚡", layout="centered")

st.title("⚡ Advanced Transformer NLP AI")
st.caption("Powered by DistilBERT trained on the public SST-2 (Stanford Sentiment Treebank) dataset.")

# 1. Load the Transformer Model (Cached so it downloads once and stays in memory)
@st.cache_resource
def load_advanced_model():
    # Downloads a neural network architecture with ~66M parameters
    # Trained on thousands of real-world sentence evaluations
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        return_all_scores=True
    )
    return classifier

with st.spinner("Initializing neural transformer weights..."):
    nlp_engine = load_advanced_model()

# 2. Interactive Input
st.subheader("Run Inference")
sample_text = st.text_area(
    "Enter complex text (slang, sarcasm, or nuanced sentences):",
    "I expected this to be a complete failure, but against all odds it blew me away."
)

if st.button("Analyze with Deep Learning"):
    if sample_text.strip():
        # Step A: Run through Tokenizer + Self-Attention Layers
        with st.spinner("Processing self-attention layers..."):
            results = nlp_engine(sample_text)[0]
        
        # Step B: Extract predictions
        # DistilBERT outputs 'POSITIVE' and 'NEGATIVE' probabilities
        scores = {item['label']: item['score'] for item in results}
        pos_score = scores.get("POSITIVE", 0.0)
        neg_score = scores.get("NEGATIVE", 0.0)

        # Step C: Render results
        if pos_score > neg_score:
            st.success(f"**Prediction:** POSITIVE (Score: {pos_score * 100:.2f}%)")
        else:
            st.error(f"**Prediction:** NEGATIVE (Score: {neg_score * 100:.2f}%)")

        st.write("#### Neural Confidence Distribution:")
        st.progress(float(pos_score), text=f"Positive: {pos_score * 100:.1f}%")
        st.progress(float(neg_score), text=f"Negative: {neg_score * 100:.1f}%")

        # Step D: Inspect why this is advanced
        with st.expander("Why is this more advanced than basic models?"):
            st.markdown("""
            * **Context-aware:** It understands that *"not bad"* is positive, whereas older models treat *"not"* and *"bad"* as negative words.
            * **Attention Mechanism:** Words dynamically influence each other across the entire sentence simultaneously.
            * **Pre-trained on Billions of Words:** The base model already knows English grammar, vocabulary, and nuance before fine-tuning on public benchmark datasets.
            """)
    else:
        st.warning("Please type a phrase to test.")
