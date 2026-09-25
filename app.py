import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="centered")

st.title("💬 24/7 Generative AI Chat")
st.caption("Powered by Google's instruction-tuned Flan-T5 model running directly in Python.")

# 1. Load the generative text2text pipeline
@st.cache_resource
def load_chat_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=100
    )

with st.spinner("Waking up chat brain..."):
    generator = load_chat_model()

# 2. Maintain conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your custom AI. Ask me anything or say hi!"}
    ]

# 3. Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle new chat messages
if user_prompt := st.chat_input("Say something..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # We format the prompt cleanly for an instruction model
            formatted_prompt = f"Answer the following message helpfully: {user_prompt}"
            output = generator(formatted_prompt)[0]["generated_text"]
            st.markdown(output)

    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": output})
