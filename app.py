import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="centered")

st.title("💬 24/7 Generative AI Chat")
st.caption("Powered by Google's Flan-T5 model running directly in Python.")

# 1. Load Tokenizer & Model directly (No pipeline KeyError issues)
@st.cache_resource
def load_chat_brain():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Waking up chat brain..."):
    tokenizer, model = load_chat_brain()

# 2. Maintain conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI assistant. Ask me anything!"}
    ]

# 3. Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle incoming user query
if user_prompt := st.chat_input("Say something..."):
    # Display user input
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate model response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare input prompt
            input_text = f"Answer the following question or respond naturally: {user_prompt}"
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate tokens
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    repetition_penalty=1.2,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50
                )
            
            # Decode tokens back into readable English
            response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

            if not response.strip():
                response = "I'm not sure how to answer that yet, try asking another question!"

            st.markdown(response)

    # Save assistant reply to memory
    st.session_state.messages.append({"role": "assistant", "content": response})
